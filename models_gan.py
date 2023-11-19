# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:01:24 2023

@author: dohyeon
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # 입력: Z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 상태: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 상태: 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 상태: 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 상태: 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
            # 최종 상태: 3 x 32 x 32
        )

    def forward(self, z):
        return self.model(z)

# 판별자(Discriminator) 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 입력: 3 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태: 64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태: 128 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256*4*4, 1)
        )

    def forward(self, x):
        return self.model(x)

def cal_gradient(t_gradients):

    # gradients를 재구성하여 각 배치의 데이터를 하나의 행으로 만듭니다.
    gradients_reshaped = t_gradients.view(t_gradients.size()[0], -1)
    
    # gradients의 2-노름을 계산합니다.
    gradient_norms = gradients_reshaped.norm(2, dim=1)
    
    # 그래디언트 페널티 계산
    deviations = gradient_norms - 1
    gradient_penalty = (deviations ** 2).mean()

    return gradient_penalty

def show_generated_imgs(generator, z_dim, device, num_images=64):
    # 잠재 벡터 z 생성
    z = torch.randn(num_images, z_dim, 1, 1, device=device)
    
    # 이미지 생성
    generator.eval() # 생성자를 평가 모드로 설정
    with torch.no_grad(): # 그라디언트 계산을 하지 않음
        fake_images = generator(z).detach().cpu()
    generator.train() # 생성자를 훈련 모드로 되돌림

    # 이미지의 픽셀 값을 [0, 1] 범위로 정규화
    fake_images = make_grid(fake_images, normalize=True, nrow=int(np.sqrt(num_images)))
    
    # 이미지 시각화
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(fake_images.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()

#%%
