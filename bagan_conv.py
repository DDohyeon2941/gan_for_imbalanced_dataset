# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:01:08 2023

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


from sklearn.preprocessing import StandardScaler

# 인코더 클래스
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            # 추가적인 레이어들을 여기에 추가하세요
        )

    def forward(self, x):
        return self.model(x)

# 디코더 클래스
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # 추가적인 레이어들을 여기에 추가하세요
        )

    def forward(self, x):
        return self.model(x)


def ensure_positive_definite(cov_matrix, noise_factor=1e-6):
    while True:
        eigenvalues, _ = torch.symeig(cov_matrix)
        if torch.min(eigenvalues) > 0:
            break  # The matrix is positive definite, break the loop
        cov_matrix += torch.eye(cov_matrix.size(0)) * noise_factor
        noise_factor *= 10  # Increase the noise factor for the next iteration if needed
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
    def __init__(self, num_classes, latent_dim):
        super(LatentVectorGenerator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        self.means = torch.zeros(num_classes, latent_dim)
        self.covariances = torch.zeros(num_classes, latent_dim, latent_dim)
        self.X_cs = []

    def fit(self, X, labels):
        epsilon = 1e-5  # A small value to avoid zero eigenvalues
        
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

    encoder = encoder.to(device)
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # inputs와 targets를 GPU로 옮김
        #inputs = inputs.view(inputs.size(0), -1)
        with torch.no_grad():
            z = encoder(inputs)
            z = z.view(z.size(0), -1)
        latent_vectors.append(z.cpu().numpy())
        labels.append(targets.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)

    latent_vector_generator.fit(torch.tensor(latent_vectors), torch.tensor(labels))

class Discriminator(nn.Module):
    def __init__(self, encoder, num_classes):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=encoder.model[0].in_channels,
                      out_channels=encoder.model[0].out_channels,
                      kernel_size=encoder.model[0].kernel_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(encoder.model[0].out_channels, 64, kernel_size=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(30976, num_classes+1),
            #nn.Linear(64 * 4 * 4, num_classes+1),
        )
        
        # 인코더의 첫 번째 Conv2D 레이어의 가중치를 복사
        self.model[0].weight.data = encoder.model[0].weight.data.clone()
        self.model[0].bias.data = encoder.model[0].bias.data.clone()

    def forward(self, x):
        #output = self.model(x)
        for layer in self.model:
            x = layer(x)
            print(x.shape)

        output = nn.functional.softmax(x, dim=1)
        return output


class Generator(nn.Module):
    def __init__(self, decoder, latent_vector_generator):
        super(Generator, self).__init__()

        self.latent_vector_generator = latent_vector_generator
        self.decoder = decoder  # 디코더 구조와 가중치를 그대로 사용
        
    def forward(self, labels):
        z = self.latent_vector_generator.sample(labels)
        z = z.to(labels.device).float()  # Ensure z is on the correct device
        z = z.squeeze(1)
        z = z.view(z.size(0), -1, 7, 7)
        return self.decoder(z)
#%%

lr = 0.0001
num_epochs = 2
input_dim = 28 * 28
output_dim = 50
batch_size = 64
#device = torch.device('cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


autoencoder = nn.Sequential(Encoder(), Decoder()).to(device)
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
#Initialize the GAN with the trained autoencoder weights


lr = 0.0002

num_classes = 10
latent_dim = 6272


# LatentVectorGenerator 준비
latent_vector_generator = LatentVectorGenerator(num_classes=num_classes, latent_dim=latent_dim).to(device)

# LatentVectorGenerator 학습
train_latent_vector_generator(autoencoder[0], latent_vector_generator, data_loader, device)

discriminator = Discriminator(autoencoder[0], num_classes).to(device)
#generator = Generator(autoencoder[1], latent_dim, num_classes).to(device)

generator = Generator(autoencoder[1], latent_vector_generator).to(device)

#discriminator = Discriminator(autoencoder[0].state_dict(), strict=False)
#generator = Generator(autoencoder[1].state_dict(), strict=False)


d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()
criterion1 = nn.CrossEntropyLoss()

###
num_classes_with_fake = num_classes + 1
fake_ratio = 1 / (num_classes_with_fake)  # 가짜 이미지 비율
num_fake_images = int(batch_size * fake_ratio)  # 생성할 가짜 이미지 수

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
        fake_labels = torch.randint(0, num_classes, (imgs.size(0),)).to(device)
        fake_images = generator(fake_labels).to(device)
        outputs = discriminator(fake_images)
        g_loss = criterion1(outputs, fake_labels)  # 판별자가 실제 라벨로 분류하도록 손실 계산

        # 생성자 업데이트
        g_loss.backward()
        g_optimizer.step()

    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

#%%


# 학습된 생성자를 이용해서 가짜 샘플 생성
import matplotlib.pyplot as plt


with torch.no_grad():
    #z = torch.randn(batch_size, latent_dim).to(device)  # 64개의 랜덤 벡터 생성
    gen_labels = torch.LongTensor(np.random.randint(0, 9, batch_size)).to(device)
    fake_images = generator(gen_labels).detach().cpu()
    fake_images = fake_images.reshape(fake_images.shape[0], 1, 28, 28)

# 이미지 출력
fig, axes = plt.subplots(1, 8, figsize=(20, 2))
for i in range(8):
    axes[i].imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
    axes[i].axis('off')
plt.show()
print(gen_labels[:8])








