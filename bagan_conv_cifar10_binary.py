# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:05:21 2023

@author: dohyeon
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models_binary_dataloader import BinaryCIFAR10Dataset
from bagan_conv_cifar10 import Encoder, Decoder, LatentVectorGenerator, train_latent_vector_generator, Discriminator, Generator, show_generated_imgs_binary




#%%
if __name__ == '__main__':
    lr = 0.0001
    num_epochs = 10
    input_dim = 32 * 32
    output_dim = 128
    batch_size = int(128/2)
    #device = torch.device('cpu')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    data_loader = DataLoader(BinaryCIFAR10Dataset(dataset, transform=None, majority_class=1, minority_class=9), batch_size=batch_size, shuffle=True, drop_last=True)

    
    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder = nn.DataParallel(autoencoder)
    autoencoder.to(device)
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    ae_loss_func = nn.MSELoss()
    
    
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(data_loader):
            #imgs = imgs.view(imgs.size(0), -1).to(device)
            imgs = imgs.to(device)
            ae_optimizer.zero_grad()
            recon = autoencoder(imgs)
            ae_loss = ae_loss_func(recon, imgs)
            ae_loss.backward()
            ae_optimizer.step()
        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [AE loss: {ae_loss.item()}] ")
    
    #%%
    
    lr = 0.0001
    num_epochs = 2
    num_classes = 2
    latent_dim = 256*4*4
    batch_size = 128 / 2

    # LatentVectorGenerator 준비
    latent_vector_generator = LatentVectorGenerator(num_classes=num_classes, latent_dim=latent_dim, device=device).to(device)
    
    # LatentVectorGenerator 학습
    train_latent_vector_generator(autoencoder.module[0], latent_vector_generator, data_loader, device)
    
    #discriminator = Discriminator(autoencoder[0], num_classes).to(device)
    #generator = Generator(autoencoder[1], latent_vector_generator).to(device)
    
    discriminator = nn.DataParallel(Discriminator(autoencoder.module[0], num_classes)).to(device)
    generator = nn.DataParallel(Generator(autoencoder.module[1], latent_vector_generator)).to(device)
    
    
    
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    
    
    criterion = nn.BCELoss()
    criterion1 = nn.CrossEntropyLoss()
    
    ###
    num_classes_with_fake = num_classes + 1
    fake_ratio = 1 / (num_classes_with_fake)  # 가짜 이미지 비율
    num_fake_images = int(batch_size * fake_ratio)  # 생성할 가짜 이미지 수
    
    loss_dict = {'d_loss':[], 'g_loss':[]}
    
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            #imgs = imgs.view(imgs.size(0), -1).to(device)
            imgs = imgs.to(device)
            labels = labels.to(device)
    
            # 판별자 훈련
            d_optimizer.zero_grad()
    
            # 실제 이미지에 대한 손실 계산
            outputs = discriminator(imgs)
            real_loss = criterion1(outputs, labels)
    
    
            # 가짜 이미지 생성 및 손실 계산
            fake_labels = torch.randint(0, num_classes, (num_fake_images,)).to(device)
            fake_images = generator(fake_labels).to(device)
            outputs = discriminator(fake_images.detach())
            fake_loss = criterion1(outputs, torch.full((len(fake_images),), num_classes).to(device))  # 가짜 라벨
    
            # 판별자 업데이트
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()
    
            # 생성자 훈련
            g_optimizer.zero_grad()
    
            # 가짜 이미지에 대한 손실 계산
            fake_labels = torch.randint(0, num_classes, (num_fake_images,)).to(device)
            fake_images = generator(fake_labels).to(device)
            outputs = discriminator(fake_images.detach())
            g_loss = criterion1(outputs, fake_labels)  # 판별자가 실제 라벨로 분류하도록 손실 계산
    
            # 생성자 업데이트
            g_loss.backward()
            g_optimizer.step()
    
            loss_dict['d_loss'].append(d_loss.item())
            loss_dict['g_loss'].append(g_loss.item())
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            if i % 10 == 0:
                show_generated_imgs_binary(generator,num_classes=num_classes, device=device)









