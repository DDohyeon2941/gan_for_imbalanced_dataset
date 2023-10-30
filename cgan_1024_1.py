# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:35:05 2023

@author: dohyeon
"""

import json
import os
import numpy as np
import csv
#import easydict
#import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep
from torchvision import datasets


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
])


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
#%%
# 소수 범주 (label=1) 만 필터링
minority_data = [(data, label) for data, label in train_dataset if label in [0,1]]

train_loader = torch.utils.data.DataLoader(minority_data,
    batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    
    
def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    """Generator, 논문에 따르면 100개의 noise를 hypercube에서 동일한 확률값으로 뽑고
       z를 200개, y를 1000개의 뉴런으로 전달합니다. 이후 1200차원의 ReLU layer로 결합하고
       Sigmoid를 통해 숫자를 만들어냅니다."""
    def __init__(self):
        super().__init__()
        self.num_classes = 2 # 클래스 수, 10
        self.nz = 100 # 노이즈 수, 100
        self.input_size = (1,28,28)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

        # # noise와 label을 결합할 용도인 label embedding matrix를 생성합니다.
        self.label_emb = nn.Embedding(2, 2)
        # 임베딩 파라미터를 선언한 후 forward 메소드를 수행하면 (입력차원, 임베딩차원) 크기를 가진 텐서가 출력
        # 이때 forward 메소드의 입력텐서는 임베딩 벡터를 추출할 범주의 인덱스이므로 무조건 정수타입(LongTensor)이 들어가야된다.
        # ex) forward 메소드에 (2,4) 크기의 텐서가 입력으로 들어가면 (2,4,10) 크기의 텐서가 출력
        # https://hongl.tistory.com/244
        
        self.model = nn.Sequential(
            nn.Linear(102, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        z = input.view(input.size(0), 100)  # 노이즈 (batch_size, 100)
        c = self.label_emb(label)  # 라벨 (10,10)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)



class Discriminator(nn.Module):
    """Discriminator, 논문에 따르면 maxout을 사용하지만
       여기서는 그냥 Fully-connected와 LeakyReLU를 사용하겠습니다.
       논문에서는 Discriminator의 구조는 그렇게 중요하지 않다고 말합니다"""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(2, 2)
        # 벡터화의 한 과정
        
        self.model = nn.Sequential(
            nn.Linear(786, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = input.view(input.size(0), 784)
        c = self.label_emb(label)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
#%%
#torch.cuda.device_count()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().cuda()
generator = nn.DataParallel(generator).to(device)
discriminator = Discriminator().cuda()
discriminator = nn.DataParallel(discriminator).to(device)

#generator = Generator().to(device)
#discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.module.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.module.parameters(), lr=1e-4)


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)

    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 2, batch_size))).to(device)  # 원-핫 벡터 아님
    fake_images = generator(z, fake_labels)  # 가짜 이미지 생성
    
    validity = discriminator(fake_images, fake_labels)  
    # discriminator에 가짜 이미지를 넣어서 결과를 출력. 가짜가 라벨과 같다라고 하면 1, 아니면 0
    
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    # dis에서 나온 출력을 1로 채워져 있는 label과 비교.
    # 만약 dis가 가짜 이미지를 라벨과 같다 판단해서 1을 출력했다면은 loss는 0에 가까워진다.
    # 이를 generator가 학습. 
    # 즉, generator는 discriminator를 잘 속이는 방향(validity 가 1이 나오게)으로 학습
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()
    
    
    
def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    # 진짜 이미지와 label을 discriminator에 넣는다.
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    # D가 진짜 이미지를 진짜 라고 맞추면 1을 출력하고 real_loss는 0이 됨
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).to(device)  # 임의의 Noise 생성
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 2, batch_size))).to(device)
    
    fake_images = generator(z, fake_labels) # Noise와 임의의 라벨을 input으로 넣음
    
    fake_validity = discriminator(fake_images, fake_labels)
    # generator가 생성한 이미지와 임의의 라벨을 discriminator에 넣어서 결과 출력 (0~1)
    
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    # real_loss를 계산할 때와는 다르게 torch.zeors를 사용해서 0으로 채워진 label을 줌.
    # discriminator가 img와 label이 같다고 판단하면 1을 출력, 아니면 0을 출력하므로 이를 CE하면
    # discriminator가 잘 맞췄을 때는 cross_entropy(1, 0) 이므로 fake_loss가 커짐
    # ---> discriminator가 진짜 이미지가 아닌 generator가 생성한 가짜 이미지를 진짜라고 판단했으므로 loss가 커짐.
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()
 
 
#%%
from torchvision.utils import make_grid

num_epochs = 100
n_critic = 5
display_step = 50
batch_size = 128
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch), end=' ')
    for i, (images, labels) in enumerate(train_loader):
        
        step = epoch * len(train_loader) + i + 1
        real_images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        generator.train()
        
        d_loss = discriminator_train_step(len(real_images), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_images, labels)
        

        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
        #writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)

        if step % display_step == 0:
            generator.eval()
            z = Variable(torch.randn(2, 100)).to(device)
            labels = Variable(torch.LongTensor(np.arange(2))).to(device)
            sample_images = generator(z, labels).unsqueeze(1)
            grid = make_grid(sample_images, nrow=3, normalize=True)
         #   writer.add_image('sample_image', grid, step)
    print('g_loss :', g_loss, 'd_loss :', d_loss)

    #print('Done!')

#%%
z = Variable(torch.randn(2, 100)).to(device)
labels = Variable(torch.LongTensor(np.arange(2))).to(device)

images = generator(z, labels).unsqueeze(1)

grid = make_grid(images, nrow=2, normalize=True)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid.permute(1, 2, 0).data.cpu(), cmap='binary')
ax.axis('off')




