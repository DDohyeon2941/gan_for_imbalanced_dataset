# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:02:56 2023

@author: dohyeon
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
minority_data = [(data, label) for data, label in train_dataset if label in [4,]][:1000]
minority_loader = DataLoader(minority_data, batch_size= 32, shuffle=True, drop_last=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
            )
    def forward(self, z):
        #z = z.view(z.size(0), 100)
        #c = self.label_emb(label)
        #x = torch.cat([z,c], 1)
        out = self.model(z)
        return out.view(z.size(0), 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.label_emb = nn.Embedding(5,2)
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        #c = self.label_emb(label)
        #x = torch.cat([x, c], 1)
        out = self.model(x)
        return out

#%%

# 하이퍼파라미터
z_dim = 100
img_dim = 784
lr = 0.0001
batch_size = 32
num_epochs = 100
clip_value = 0.01


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')


# 모델 및 옵티마이저 초기화
generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)


###

for epoch in range(num_epochs):
    for real_data, _ in minority_loader:
        real_data = real_data.view(real_data.size(0),-1).to(device)

        # 판별자 학습
        for _ in range(5):
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(z)

            d_loss_real = discriminator(real_data)
            d_loss_fake = discriminator(fake_images)

            d_loss = -torch.mean(d_loss_real) + torch.mean(d_loss_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

        #생성자 학습
        g_optimizer.zero_grad()

        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)
        output = discriminator(fake_images)

        g_loss = -torch.mean(output)

        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}] D Loss: {d_loss.item()} G Loss: {g_loss.item()}")



#%%



# 학습된 생성자를 이용해서 가짜 샘플 생성
import matplotlib.pyplot as plt

with torch.no_grad():
    z = torch.randn(32, z_dim).to(device)  # 64개의 랜덤 벡터 생성
    #gen_labels = torch.LongTensor(np.random.randint(4, 5, z.shape[0])).to(device)
    fake_images = generator(z).detach().cpu()

# 이미지 출력
fig, axes = plt.subplots(1, 8, figsize=(20, 2))
for i in range(8):
    axes[i].imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
    axes[i].axis('off')
plt.show()
#print(gen_labels[:8])










