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


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim*2)
            )


    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim*2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
            )

    def forward(self, x):

        return self.model(x)
#%%

class LatentVectorGenerator(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(LatentVectorGenerator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # means and covariances should be PyTorch tensors
        self.means = torch.zeros(num_classes, latent_dim)
        self.covariances = torch.zeros(num_classes, latent_dim, latent_dim)

    def fit(self, X, labels):
        for c in range(self.num_classes):
            indices = (labels == c).nonzero()
            # nonzero() 함수가 반환하는 값의 형태에 따라 적절하게 인덱싱
            if isinstance(indices, tuple):
                indices = indices[0]
            X_c = torch.tensor(X[indices])
            mean_c = torch.mean(X_c, axis=0)
            cov_c = torch.tensor(np.cov(X_c, rowvar=False))
            # 공분산 행렬의 대각선에 작은 값을 추가
            cov_c += torch.eye(self.latent_dim) * 1e-6
            self.means[c] = mean_c
            self.covariances[c] = cov_c

    def sample(self, labels):
        if np.isscalar(labels):
            labels = [labels,]
        z = []
        for c in labels:
            mean = self.means[c]
            cov = self.covariances[c]
           # MultivariateNormal로부터 샘플링하는 동안 그라디언트를 계산하도록 설정
            mean = mean.clone().detach().requires_grad_(True)  
            cov = cov.clone().detach().requires_grad_(True)
            
            sample = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample()
            z.append(sample)
        return torch.stack(z)

def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows
            contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # Uncomment this line if m is not a double tensor
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

# Encoder를 사용하여 잠재 벡터를 얻고, 이를 LatentVectorGenerator에 학습시킵니다.
def train_latent_vector_generator(encoder, latent_vector_generator, data_loader, device):
    latent_vectors = []
    labels = []

    encoder = encoder.to(device)
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # inputs와 targets를 GPU로 옮김
        inputs = inputs.view(inputs.size(0), -1)
        with torch.no_grad():
            z = encoder(inputs)
        latent_vectors.append(z.cpu().numpy())
        labels.append(targets.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)
    
    latent_vector_generator.fit(latent_vectors, labels)

class Discriminator(nn.Module):
    def __init__(self, encoder, num_classes):
        super(Discriminator, self).__init__()
        
        # 인코더의 구조를 그대로 사용하고 가중치를 복사
        self.model = nn.Sequential(
            nn.Linear(encoder.model[0].in_features, encoder.model[0].out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(encoder.model[2].in_features, encoder.model[2].out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(encoder.model[2].out_features, num_classes+1),
        )
        
        # 인코더의 가중치를 복사
        self.model[0].weight.data = encoder.model[0].weight.data.clone()
        self.model[0].bias.data = encoder.model[0].bias.data.clone()
        self.model[2].weight.data = encoder.model[2].weight.data.clone()
        self.model[2].bias.data = encoder.model[2].bias.data.clone()

    def forward(self, x):
        output = self.model(x)
        output = nn.functional.softmax(output, dim=1)
        return output
    
class Generator(nn.Module):
    def __init__(self, latent_vector_generator, decoder, latent_dim, num_classes):
        super(Generator, self).__init__()

        self.latent_vector_generator = latent_vector_generator
        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.model = decoder # assuming the decoder is passed as an argument

    def forward(self, labels):
        # Sample a latent vector based on the class labels
        z = []
        for c in labels.cpu().numpy():
            z_c = self.latent_vector_generator.sample(c)
            z.append(z_c)
        #z = torch.tensor(z, dtype=torch.float32).to(labels.device)
        print(z)
        if isinstance(z, torch.Tensor):
            z = z.to(labels.device).float()
        embedded_labels = self.embedding(labels)
        input_vector = torch.cat([z, embedded_labels], dim=1)

        return self.model(input_vector)



class Generator1(nn.Module):
    def __init__(self, decoder, latent_vector_generator, num_classes, latent_dim):
        super(Generator1, self).__init__()

        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.latent_vector_generator = latent_vector_generator
        # 디코더의 구조를 그대로 사용하고 가중치를 복사
        self.model = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, decoder.model[0].out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(decoder.model[0].out_features, decoder.model[2].out_features),
            nn.Sigmoid()
        )
        
        # 디코더의 가중치를 복사
        self.model[0].weight.data = decoder.model[0].weight.data.clone()
        self.model[0].bias.data = decoder.model[0].bias.data.clone()
        self.model[2].weight.data = decoder.model[2].weight.data.clone()
        self.model[2].bias.data = decoder.model[2].bias.data.clone()

    def forward(self, labels):
        labels_np = labels.cpu().numpy()
        z = []
        for label in labels_np:
            z_sample = self.latent_vector_generator.sample(label)
            z.append(z_sample)
        #z = torch.tensor(z, device=labels.device).float()
        print(type(z))
        # z가 텐서의 리스트인 경우
        if isinstance(z, list) and all(isinstance(zi, torch.Tensor) for zi in z):
            z = torch.stack(z).to(labels.device).float()
        print(type(z))

        embedded_labels = self.embedding(labels)
        z = z.squeeze(1)
        print(f"Shape of z: {z.shape}")
        print(f"Shape of embedded_labels: {embedded_labels.shape}")

        input_vector = torch.cat([z, embedded_labels], dim=1)

        print("Shape of input_vector:", input_vector.shape)

        # 첫 번째 Linear layer의 weight shape를 출력
        print("Shape of the first layer weight:", self.model[0].weight.shape)


        return self.model(input_vector)
#%%

lr = 0.0001
num_epochs = 2
input_dim = 28 * 28
output_dim = 64
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


autoencoder = nn.Sequential(Encoder(input_dim, output_dim), Decoder(output_dim, input_dim)).to(device)
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
ae_loss_func = nn.MSELoss()


for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(data_loader):
        imgs = imgs.view(imgs.size(0), -1).to(device)
        ae_optimizer.zero_grad()
        recon = autoencoder(imgs)
        ae_loss = ae_loss_func(recon, imgs)
        ae_loss.backward()
        ae_optimizer.step()
    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [AE loss: {ae_loss.item()}] ")


#%%
#Initialize the GAN with the trained autoencoder weights

num_classes = 10
latent_dim = 64


# LatentVectorGenerator 준비
latent_vector_generator = LatentVectorGenerator(num_classes=num_classes, latent_dim=128).to(device)

# LatentVectorGenerator 학습
train_latent_vector_generator(autoencoder[0], latent_vector_generator, data_loader, device)

discriminator = Discriminator(autoencoder[0], num_classes).to(device)
#generator = Generator(autoencoder[1], latent_dim, num_classes).to(device)

generator = Generator1(autoencoder[1], latent_vector_generator, latent_dim, num_classes).to(device)

#discriminator = Discriminator(autoencoder[0].state_dict(), strict=False)
#generator = Generator(autoencoder[1].state_dict(), strict=False)


d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

criterion = nn.BCELoss()
criterion1 = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.view(imgs.size(0), -1).to(device)
        labels = labels.to(device)


        #진짜와 가짜 라벨 생성
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # 판별자 훈련
        d_optimizer.zero_grad()

        outputs = discriminator(imgs)
        real_loss = criterion1(outputs, labels)

        #가짜 이미지 생성
        #z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_images = generator(labels).to(device)

        outputs = discriminator(fake_images.detach())
        fake_loss = criterion1(outputs, torch.full((len(fake_images),), num_classes).to(device))

        #역전파와 최적화
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # 생성자 훈련
        g_optimizer.zero_grad()
        
        outputs = discriminator(fake_images)
        g_loss = criterion1(outputs, torch.full((len(fake_images),), num_classes).to(device))
        
        g_loss.backward()
        g_optimizer.step()
    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

#%%


# 학습된 생성자를 이용해서 가짜 샘플 생성
import matplotlib.pyplot as plt


with torch.no_grad():
    z = torch.randn(batch_size, latent_dim).to(device)  # 64개의 랜덤 벡터 생성
    gen_labels = torch.LongTensor(np.random.randint(0, 9, z.shape[0])).to(device)
    fake_images = generator(z, gen_labels).detach().cpu()
    fake_images = fake_images.reshape(fake_images.shape[0], 1, 28, 28)

# 이미지 출력
fig, axes = plt.subplots(1, 8, figsize=(20, 2))
for i in range(8):
    axes[i].imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
    axes[i].axis('off')
plt.show()
#print(gen_labels[:8])








