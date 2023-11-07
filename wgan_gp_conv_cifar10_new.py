# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:37:52 2023

@author: dohyeon
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import grad

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


batch_size = 128
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)

#train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, drop_last=True)

minority_data = [(data, label) for data, label in train_dataset if label in [0,]]
minority_loader = DataLoader(minority_data, batch_size= batch_size, shuffle=True, drop_last=True)
#%%

class ResidualBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=None):
        super(ResidualBlockGenerator, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()
        if upsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
    def forward(self, x):
        original_x = x  # 원본 x를 저장합니다.
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            #print("Upsampled x shape:", x.shape)
        
        out = self.block(x)

        #print('Block output shape:', out.shape)
        
        # 여기에서 원본 x를 단축 경로에 적용하기 전에 업샘플링 여부를 확인합니다.
        if self.upsample:
            shortcut = self.shortcut(original_x)  # 업샘플링 된 원본 x를 단축 경로에 적용합니다.
        else:
            shortcut = self.shortcut(x)  # 변경되지 않은 x를 단축 경로에 적용합니다.
        
        #print('Shortcut output shape:', shortcut.shape)
    
        return out + shortcut
#%%
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # First linear layer
            nn.Linear(z_dim, 128 * 4 * 4, bias=False),
            # Reshape to (batch_size, 128, 4, 4)
            View((-1, 128, 4, 4)),
            # Residual Blocks
            ResidualBlockGenerator(128, 128, upsample=True),
            ResidualBlockGenerator(128, 128, upsample=True),
            ResidualBlockGenerator(128, 128, upsample=True),
            # Output layer
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):

        #self.model(z)
        """
        for layer in self.model:
            x = layer(x)
            print(x.shape)
        """
        return self.model(x)

class ResidualBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlockDiscriminator, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.AvgPool2d(2, stride=2)
            )

    def forward(self, x):
        out = self.block(x)
        shortcut = self.shortcut(x)
        if self.downsample:
            out = nn.functional.avg_pool2d(out, 2)
        return out + shortcut

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input layer
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            ResidualBlockDiscriminator(64, 128, downsample=True),
            ResidualBlockDiscriminator(128, 128),
            ResidualBlockDiscriminator(128, 128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        #self.model(x)
        """
        for layer in self.model:
            x = layer(x)
            print(x.shape)
        """
        return self.model(x)

# Helper class to reshape tensors
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


#%%


lr = 0.0001
z_dim = 100
num_epochs = 100
image_size = 32  # Should match the output image size of the generato
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 모델 및 옵티마이저 초기화
generator = nn.DataParallel(Generator(z_dim))
generator.to(device)
discriminator = nn.DataParallel(Discriminator())
discriminator.to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(minority_loader):
        # Train Discriminator with real images
        real_images = real_images.to(device)
        discriminator.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        real_predictions = discriminator(real_images.detach())
        real_loss = criterion(real_predictions, real_labels)
        real_loss.backward()

        # Train Discriminator with fake images
        noise = torch.randn(real_images.size(0), z_dim).to(device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
        fake_predictions = discriminator(fake_images.detach())
        fake_loss = criterion(fake_predictions, fake_labels)
        fake_loss.backward()
        
        d_loss = real_loss + fake_loss
        d_optimizer.step()

        # Train Generator
        generator.zero_grad()
        trick_labels = torch.ones(real_images.size(0), 1).to(device)
        predictions = discriminator(fake_images)
        g_loss = criterion(predictions, trick_labels)
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(minority_loader)}], '
                  f'D_Loss: {d_loss.item()}, G_Loss: {g_loss.item()}')

#%%



# 학습된 생성자를 이용해서 가짜 샘플 생성
import matplotlib.pyplot as plt

with torch.no_grad():
    z = torch.randn(batch_size, z_dim).to(device)  # 64개의 랜덤 벡터 생성
    #gen_labels = torch.LongTensor(np.random.randint(4, 5, z.shape[0])).to(device)
    fake_images = generator(z).detach().cpu()

# 이미지 출력
fig, axes = plt.subplots(1, 8, figsize=(20, 2))
for i in range(8):
    axes[i].imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
    axes[i].axis('off')
plt.show()
#print(gen_labels[:8])




