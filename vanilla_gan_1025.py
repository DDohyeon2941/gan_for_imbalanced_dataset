# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:30:34 2023

@author: dohyeon
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
minority_data = [(data, label) for data, label in train_dataset if label == 8]
minority_loader = DataLoader(minority_data, batch_size= 32, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
            )
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr= 1e-4)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr= 1e-4)

num_epochs = 100
latent_size = 100

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(minority_loader):
        imgs = imgs.to(device)

        #Discriminator 학습
        optimizer_d.zero_grad() # 그래디언트 초기화
        real_labels = torch.ones(imgs.size(0), 1).to(device) # 진짜 라벨 생성
        fake_labels = torch.zeros(imgs.size(0), 1).to(device) # 가짜 라벨 생성

        outputs = discriminator(imgs) # 진짜 샘플에 대한 판별자의 예측결과 산출
        d_loss_real = criterion(outputs, real_labels) # 손실값 산출

        z = torch.randn(imgs.size(0), latent_size).to(device) # z 벡터 생성
        fake_images = generator(z) # 가짜 샘플 생성
        outputs = discriminator(fake_images.detach()) # 가짜 샘플에 대한 판별자의 예측결과 산출
        d_loss_fake = criterion(outputs, fake_labels) # 손실값 산출
        d_loss = d_loss_real + d_loss_fake # 손실값
        d_loss.backward()
        optimizer_d.step()

        #print('discriminator >>> generator')
        #Generator 학습
        optimizer_g.zero_grad() # 그래디언트 초기화
        z = torch.randn(imgs.size(0), latent_size).to(device) # z 벡터 생성
        fake_images = generator(z) # 가짜 샘플 생성
        outputs = discriminator(fake_images) # 가짜 샘플에 대한 판별자의 예측결과 산출
        g_loss = criterion(outputs, real_labels) # 손실값 산출
        g_loss.backward()
        optimizer_g.step()

        if (i+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}")

print('Training complete.')

#%%
# 학습된 생성자를 이용해서 가짜 샘플 생성
import matplotlib.pyplot as plt

with torch.no_grad():
    z = torch.randn(32, latent_size).to(device)  # 64개의 랜덤 벡터 생성
    fake_images = generator(z).detach().cpu()

# 이미지 출력
fig, axes = plt.subplots(1, 8, figsize=(20, 2))
for i in range(8):
    axes[i].imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
    axes[i].axis('off')
plt.show()

















