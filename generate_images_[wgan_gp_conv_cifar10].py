# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:31:12 2023

@author: dohyeon
"""

import os
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from wgan_gp_conv_cifar10 import Generator


def visualize_image_from_dataloader(data_loader):
    # 첫 번째 배치의 첫 번째 이미지 가져오기
    images, labels = next(iter(data_loader))
    image = images[0]

    # 이미지의 정규화를 원래 범위로 되돌리기
    image = image / 2 + 0.5  # [0, 1] 범위로 되돌리기
    image = image.numpy()  # 텐서를 NumPy 배열로 변환

    # 시각화
    plt.figure(figsize=(4, 4))
    plt.imshow(np.transpose(image, (1, 2, 0)))  # [C, H, W] -> [H, W, C]
    plt.axis('off')  # 축 레이블 숨기기
    plt.show()





def visualize_images_batch(data_loader, nrow=8):
    # 첫 번째 배치 가져오기
    images, labels = next(iter(data_loader))

    # 이미지 그리드 생성
    grid = make_grid(images, nrow=nrow, normalize=True)


    # 이미지의 차원변경
    np_grid = grid.numpy()

    # 시각화
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(np_grid, (1, 2, 0)))  # [C, H, W] -> [H, W, C]
    plt.axis('off')
    plt.show()






#%%
# 생성자 모델을 로드합니다.

z_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = nn.DataParallel(Generator(z_dim))
generator.to(device)


generator.load_state_dict(torch.load('generator_prop_imb.pth'))

# 이미지 생성 및 저장
num_images = 1000  # 생성할 이미지 수

generator.eval() # 생성자를 평가 모드로 설정
with torch.no_grad(): # 그라디언트 계산을 하지 않음
    z = torch.randn(num_images, z_dim, 1, 1, device=device)
    generated_images = generator(z).detach().cpu()



torch.save(generated_images, 'saved_tensor.pt')
#loaded_images = torch.load('saved_tensor.pt')





#%%
# 이미지 저장 경로
save_dir = './generated_images/minority_cifar10_5/'
os.makedirs(save_dir, exist_ok=True)

# 생성된 이미지 저장
for i, image in enumerate(generated_images):
    save_image(image, os.path.join(save_dir, f'generated_image_{i}.png'))





#%%

# 저장된 이미지를 로드하기 위한 데이터셋 정의
dataset = datasets.ImageFolder(
    root='./generated_images/',
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

# 데이터 로더 설정
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 사용 예시: 데이터 로더를 통해 이미지와 레이블 배치 로드
for images, labels in data_loader:
    # 여기서 이미지와 레이블을 사용할 수 있습니다.
    pass



visualize_image_from_dataloader(data_loader)


visualize_images_batch(data_loader)

#%%

import torchvision
# 이미지 출력
fig, axes = plt.subplots(8, 8, figsize=(20, 20))  # 8x8 격자로 설정
for i, ax in enumerate(axes.flat):
    # torchvision.utils.make_grid를 사용하여 이미지 정규화 및 그리드 생성
    ax.imshow(torchvision.utils.make_grid(generated_images[i], normalize=True).permute(1, 2, 0))
    ax.axis('off')
plt.show()


#%%

