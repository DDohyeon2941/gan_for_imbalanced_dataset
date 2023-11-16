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
from torchvision.utils import make_grid
from wgan_gp_conv_cifar10 import Generator, Discriminator, cal_gradient, show_generated_imgs





#%%
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    batch_size = 64
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)

    #train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, drop_last=True)

    minority_data = [(data, label) for data, label in train_dataset if label in [5,]]
    minority_loader = DataLoader(minority_data, batch_size= batch_size, shuffle=True, drop_last=True)



    # 하이퍼파라미터
    z_dim = 100
    img_dim = 32*32
    lr = 0.0001
    #batch_size = 128
    num_epochs = 100
    lambda1 = 20
    lambda2 = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #device = torch.device('cpu')
    
    
    # 모델 및 옵티마이저 초기화
    generator = nn.DataParallel(Generator(z_dim))
    generator.to(device)
    discriminator1 = nn.DataParallel(Discriminator())
    discriminator1.to(device)
    
    
    
    discriminator2 = nn.DataParallel(Discriminator())
    discriminator2.load_state_dict(torch.load('discriminator.pth'))
    discriminator2.to(device)
    
    
    g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
    d_optimizer1 = optim.RMSprop(discriminator1.parameters(), lr=lr)
    #d_optimizer2 = optim.RMSprop(discriminator2.parameters(), lr=lr)
    
    
    ###
    loss_dict = {'d_loss':[], 'g_loss':[]}
    
    for epoch in range(num_epochs):
        for real_data, _ in minority_loader:
            #real_data = real_data.view(real_data.size(0),-1).to(device)
            real_data = real_data.to(device)
            # 판별자 학습
            for _ in range(5):
                d_optimizer1.zero_grad()
                #d_optimizer2.zero_grad()
    
    
                z = torch.randn(batch_size, z_dim,1,1).to(device)
                fake_images = generator(z)
    
                d_loss_real = discriminator1(real_data)
                d_loss_fake_min = discriminator1(fake_images)
                d_loss_fake_maj = discriminator2(fake_images)
    
                alpha = torch.rand(batch_size, 1,1,1).to(device)
                alpha.to(device)
    
                x_hat = (alpha * real_data + (1 - alpha) * fake_images).detach()
                x_hat.requires_grad = True
    
                pred_hat_min = discriminator1(x_hat)
                #pred_hat_maj = discriminator2(x_hat)
    
                gradients_min = grad(outputs=pred_hat_min, inputs=x_hat, grad_outputs=torch.ones(pred_hat_min.size()).to(device),
                                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
    
                """
                gradients_maj = grad(outputs=pred_hat_maj, inputs=x_hat, grad_outputs=torch.ones(pred_hat_maj.size()).to(device),
                                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                """
                gradient_penalty_min = cal_gradient(gradients_min, lambda1)
                #gradient_penalty_maj = cal_gradient(gradients_maj, lambda1)
    
                #output_diff = torch.abs(d_loss_fake_min  - d_loss_fake_maj)
    
                d_loss = -torch.mean(d_loss_real) + (1*torch.mean(d_loss_fake_min)) + (10* gradient_penalty_min) + (0*torch.mean(d_loss_fake_maj))
    
                #d_loss = 0.6 * (-torch.mean(d_loss_real) + torch.mean(d_loss_fake_min) + gradient_penalty_min) + 0.4 * torch.mean(d_loss_fake_maj)
    
                d_loss.backward()
                d_optimizer1.step()
    
    
            #생성자 학습
            g_optimizer.zero_grad()
    
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = generator(z)
            output_min = discriminator1(fake_images)
            output_maj = discriminator2(fake_images)
    
            g_loss = -(1*torch.mean(output_min)) - (0*torch.mean(output_maj))
    
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















