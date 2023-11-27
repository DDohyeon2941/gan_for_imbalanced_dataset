# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:53:33 2023

@author: dohyeon
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from models_binary_dataloader import BinaryCIFAR10Dataset
import torch.nn.functional as F



# 인코더 클래스
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)

            # 추가적인 레이어들을 여기에 추가하세요
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, initial_size=8, edge_dim=32):
        super(Decoder, self).__init__()
        self.initial_size = initial_size

        # 첫 번째 ConvTranspose2d는 잠재 벡터를 처리
        self.initial_conv = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1)

        # 엣지 맵을 처리하는 Conv2d 레이어 (입력 채널 수를 3으로 변경)
        self.edge_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        # 나머지 네트워크 구조
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, edge_map):
        # 잠재 벡터 처리
        z = z.view(z.size(0), 64, self.initial_size, self.initial_size)
        z = self.initial_conv(z)

        # 엣지 맵 처리
        edge_map = edge_map.expand(-1, 3, -1, -1)  # 엣지 맵을 3채널로 확장
        edge_map = F.interpolate(edge_map, size=z.size()[2:], mode='bilinear', align_corners=False)  # 크기 조정
        edge_map = self.edge_conv(edge_map)
        #print(z.shape, edge_map.shape)
        # 잠재 벡터와 엣지 맵 결합
        combined = z + edge_map  # 여기서는 단순히 합산을 사용

        # 나머지 네트워크를 통해 이미지 생성
        img = self.model(combined)
        return img

def sobel_edge_detection(image_batch):
    # 소벨 커널 정의
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)

    # 필터를 현재 장치로 이동 (CPU 또는 CUDA)
    if image_batch.is_cuda:
        sobel_x = sobel_x.to(image_batch.device)
        sobel_y = sobel_y.to(image_batch.device)

    # 배치의 각 이미지에 대해 채널별로 소벨 필터 적용
    batch_size, channels, _, _ = image_batch.shape
    edge_maps = torch.zeros_like(image_batch)

    for i in range(batch_size):
        for j in range(channels):
            edge_x = F.conv2d(image_batch[i, j].view(1, 1, image_batch.size(2), image_batch.size(3)), sobel_x, padding=1)
            edge_y = F.conv2d(image_batch[i, j].view(1, 1, image_batch.size(2), image_batch.size(3)), sobel_y, padding=1)
            edge_maps[i, j] = torch.sqrt(edge_x**2 + edge_y**2)

    return edge_maps
# 기존의 Autoencoder 클래스 수정 또는 새로운 클래스 정의
class AutoencoderWithEdgeMap(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoencoderWithEdgeMap, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        edge_map = sobel_edge_detection(x)  # 엣지 맵 생성
        recon = self.decoder(z, edge_map)
        return recon

def ensure_positive_definite(cov_matrix, noise_factor=1e-6):
    while True:
        eigenvalues, _ = torch.symeig(cov_matrix)
        if torch.min(eigenvalues) > 0:
            break  # The matrix is positive definite, break the loop
        cov_matrix += torch.eye(cov_matrix.size(0)) * noise_factor
        noise_factor *= 0.01  # Increase the noise factor for the next iteration if needed
    #cov_matrix[cov_matrix<0] = noise_factor

    return cov_matrix

def torch_cov2(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m -= torch.mean(m, dim=1, keepdim=True)
    cov = m @ m.t() / (m.size(1) - 1)
    
    # Ensure the covariance matrix is positive definite
    cov = ensure_positive_definite(cov)

    return cov

class LatentVectorGenerator(nn.Module):
    def __init__(self, num_classes, latent_dim, device):
        super(LatentVectorGenerator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        self.means = torch.zeros(num_classes, latent_dim).to(device)
        self.covariances = torch.zeros(num_classes, latent_dim, latent_dim).to(device)
        self.X_cs = []

    def fit(self, X, labels):
        #epsilon = 1e-5  # A small value to avoid zero eigenvalues
        
        for c in range(self.num_classes):
            #indices = torch.tensor((labels == c).nonzero()[0])
            indices = (labels == c).nonzero().squeeze().clone().detach()
            print((labels == c).nonzero(), c, indices.shape)
            if indices.dim() > 1:  # nonzero의 출력이 2D 텐서인 경우
                indices = indices[:, 0]  # 첫 번째 차원을 선택
            
            X_c = X[indices]
            X_c = X_c.view(X_c.size(0), -1)
            self.X_cs.append(X_c)
            print(X_c.shape)
            # Calculate mean
            mean_c = torch.mean(X_c, axis=0)
            self.means[c] = mean_c
            
            # Calculate covariance matrix
            cov_c = torch_cov2(X_c)

            self.covariances[c] = cov_c

    def sample(self, labels):
        if np.isscalar(labels):
            labels = [labels,]

        z = []
        for c in labels:
            mean = self.means[c]
            cov = self.covariances[c]
            sample = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample()
            z.append(sample)
        return torch.stack(z)




# Encoder를 사용하여 잠재 벡터를 얻고, 이를 LatentVectorGenerator에 학습시킵니다.
def train_latent_vector_generator(encoder, latent_vector_generator, data_loader, device):
    latent_vectors = []
    labels = []

    #encoder = encoder.to(device)
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # inputs와 targets를 GPU로 옮김
        #inputs = inputs.view(inputs.size(0), -1)
        with torch.no_grad():
            z = encoder(inputs).to(device)
            z = z.view(z.size(0), -1)
        latent_vectors.append(z.cpu().numpy())
        labels.append(targets.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)

    latent_vector_generator.fit(torch.tensor(latent_vectors), torch.tensor(labels))



class EdgeGenerator(nn.Module):
    def __init__(self, latent_dim, edge_dim):
        super(EdgeGenerator, self).__init__()
        # 네트워크 구조 정의
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, edge_dim * edge_dim),
            nn.Sigmoid()
        )
        self.edge_dim = edge_dim

    def forward(self, z):
        edge_map = self.model(z)
        edge_map = edge_map.view(-1, 1, self.edge_dim, self.edge_dim)  # edge_dim x edge_dim 크기로 변형
        return edge_map

class Generator(nn.Module):
    def __init__(self, decoder, latent_vector_generator, edge_dim, edge_generator):
        super(Generator, self).__init__()
        self.latent_vector_generator = latent_vector_generator
        self.decoder = decoder
        self.edge_generator = edge_generator

    def forward(self, labels):
        z = self.latent_vector_generator.sample(labels)
        #print("z device:", z.device)  # 장치 확인
        edge_map = self.edge_generator(z)
        #edge_map = edge_map.to(z.device)
        #print("edge_map device:", edge_map.device)  # 장치 확인
        z = z.view(z.size(0), -1, 4, 4)
        img = self.decoder(z, edge_map)
        return img
class Discriminator(nn.Module):
    def __init__(self, encoder, num_classes):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=encoder.model[0].in_channels,
                      out_channels=encoder.model[0].out_channels,
                      kernel_size=encoder.model[0].kernel_size,
                      stride = encoder.model[0].stride,
                      padding = encoder.model[0].padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(encoder.model[0].out_channels, 128, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(64*4*4, num_classes+1), # 64*22*22
            #nn.Linear(64 * 4 * 4, num_classes+1),
        )

        # 인코더의 첫 번째 Conv2D 레이어의 가중치를 복사
        self.model[0].weight.data = encoder.model[0].weight.data.clone()
        self.model[0].bias.data = encoder.model[0].bias.data.clone()

    def forward(self, x):
        x = self.model(x)
        """
        for layer in self.model:
            x = layer(x)
            print(x.shape)
        """
        #output = nn.functional.softmax(x, dim=1)
        return x

def show_generated_imgs_binary(generator, device, num_classes=2, num_images=64):
    # 잠재 벡터 z 생성
    z = torch.LongTensor(np.random.randint(0, num_classes, num_images)).to(device)
    
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
if __name__ == '__main__':
    lr = 0.0001
    num_epochs = 15
    input_dim = 32 * 32
    output_dim = 128
    batch_size = int(128 / 2)
    #device = torch.device('cpu')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    data_loader = DataLoader(BinaryCIFAR10Dataset(dataset, transform=None, majority_class=1, minority_class=3), batch_size=batch_size, shuffle=True, drop_last=True)

    
    autoencoder = AutoencoderWithEdgeMap(Encoder(), Decoder()).to(device)
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    ae_loss_func = nn.MSELoss()

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(data_loader):
            imgs = imgs.to(device)
            ae_optimizer.zero_grad()
            recon = autoencoder(imgs)  # 엣지 맵이 내부적으로 생성되고 사용됨
            ae_loss = ae_loss_func(recon, imgs)
            ae_loss.backward()
            ae_optimizer.step()
        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [AE loss: {ae_loss.item()}] ")

    #%%
    lr = 0.0001
    num_epochs = 3
    num_classes = 2
    latent_dim = 256*4*4
    batch_size = 128 / 2
    edge_dim= 32

    # LatentVectorGenerator 준비
    latent_vector_generator = LatentVectorGenerator(num_classes=num_classes, latent_dim=latent_dim, device=device).to(device)
    
    # LatentVectorGenerator 학습
    train_latent_vector_generator(autoencoder.encoder, latent_vector_generator, data_loader, device)
    
    #discriminator = Discriminator(autoencoder[0], num_classes).to(device)
    #generator = Generator(autoencoder[1], latent_vector_generator).to(device)

    edge_generator = EdgeGenerator(latent_vector_generator.latent_dim, edge_dim)

    discriminator = Discriminator(autoencoder.encoder, num_classes).to(device)
    generator = Generator(autoencoder.decoder, latent_vector_generator,edge_dim=edge_dim, edge_generator=edge_generator).to(device)
    
    
    
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





















