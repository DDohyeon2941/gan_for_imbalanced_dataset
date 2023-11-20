# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:40:35 2023

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
from models_gan import Generator, Discriminator, cal_gradient, show_generated_imgs


#%%
if __name__ == '__main__':

    # 다수범주 기반 생성자 및 판별자 학습 (3)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    majority_class = 1
    batch_size = 128
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    
    #train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, drop_last=True)
    
    minority_data = [(data, label) for data, label in train_dataset if label in [majority_class,]]
    minority_loader = DataLoader(minority_data, batch_size= batch_size, shuffle=True, drop_last=True)


    # 하이퍼파라미터
    z_dim = 128
    img_dim = 32*32
    lr = 0.0001
    num_epochs = 300

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #device = torch.device('cpu')
    
    
    # 모델 및 옵티마이저 초기화
    generator = nn.DataParallel(Generator(z_dim))
    generator.to(device)
    discriminator = nn.DataParallel(Discriminator())
    discriminator.to(device)
    
    
    
    g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)
    
    
    ###
    loss_dict = {'d_loss':[], 'g_loss':[]}
    
    for epoch in range(num_epochs):
        for real_data, _ in minority_loader:
            #real_data = real_data.view(real_data.size(0),-1).to(device)
            real_data = real_data.to(device)
            # 판별자 학습
            for _ in range(5):
                d_optimizer.zero_grad()
    
                z = torch.randn(batch_size, z_dim,1,1).to(device)
                fake_images = generator(z)
    
                d_loss_real = discriminator(real_data)
                d_loss_fake = discriminator(fake_images)
    
                alpha = torch.rand(batch_size, 1,1,1).to(device)
                alpha.to(device)
    
                x_hat = (alpha * real_data + (1 - alpha) * fake_images).detach()
                x_hat.requires_grad = True
    
                pred_hat = discriminator(x_hat)
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
    
                gradient_penalty = cal_gradient(gradients)
    
                d_loss = -torch.mean(d_loss_real) + torch.mean(d_loss_fake) + (10*gradient_penalty)
    
                d_loss.backward()
                d_optimizer.step()
    
    
            #생성자 학습
            g_optimizer.zero_grad()
    
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = generator(z)
            output = discriminator(fake_images)
    
            g_loss = -torch.mean(output)
    
            g_loss.backward()
            g_optimizer.step()
    
    
            loss_dict['d_loss'].append(d_loss.item())
            loss_dict['g_loss'].append(g_loss.item())
    
            print(f"Epoch [{epoch + 1}/{num_epochs}] D Loss: {d_loss.item()} G Loss: {g_loss.item()}")
    
        if epoch % 10 == 0:
            show_generated_imgs(generator, z_dim, device)
    #%%
    
    # 학습된 생성자를 이용해서 가짜 샘플 생성
    import matplotlib.pyplot as plt
    
    with torch.no_grad():
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)  # 64개의 랜덤 벡터 생성
        #gen_labels = torch.LongTensor(np.random.randint(4, 5, z.shape[0])).to(device)
        fake_images = generator(z).detach().cpu()
    
    # 이미지 출력
    fig, axes = plt.subplots(1, 8, figsize=(20, 2))
    for i in range(8):
        axes[i].imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
        axes[i].axis('off')
    plt.show()
    
    #print(gen_labels[:8])
    
    #%%
    
    
    with torch.no_grad():
        z = torch.randn(64, z_dim, 1, 1).to(device)  # 64개의 랜덤 벡터 생성
        fake_images = generator(z).detach().cpu()
    
    # 이미지 출력
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))  # 8x8 격자로 설정
    for i, ax in enumerate(axes.flat):
        # torchvision.utils.make_grid를 사용하여 이미지 정규화 및 그리드 생성
        ax.imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
        ax.axis('off')
    plt.show()
    
    
    
    
    #%%
    
    
    plt.plot(loss_dict['d_loss'], c='b', label='d_loss')
    plt.plot(loss_dict['g_loss'], c='g', label='g_loss')
    plt.legend()
    
    
    
    #%%
    
    torch.save(generator.state_dict(), 'generator_cifar10_car.pth')
    torch.save(discriminator.state_dict(), 'discriminator_cifar10_car.pth')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
