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

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim)
            )


    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
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
            indices = torch.tensor((labels == c).nonzero()[0])
            if indices.dim() > 1:  # nonzero의 출력이 2D 텐서인 경우
                indices = indices[:, 0]  # 첫 번째 차원을 선택
            
            X_c = X[indices]
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
        inputs = inputs.view(inputs.size(0), -1)
        with torch.no_grad():
            z = encoder(inputs)
        latent_vectors.append(z.cpu().numpy())
        labels.append(targets.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)
    
    # Standardize the latent vectors
    scaler = StandardScaler()
    latent_vectors = scaler.fit_transform(latent_vectors)
    
    latent_vector_generator.fit(torch.tensor(latent_vectors), labels)

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
            nn.Linear(latent_dim + latent_dim, latent_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim*2, decoder.model[2].out_features),
            nn.Sigmoid()
        )
        
        # 디코더의 가중치를 복사
        self.model[0].weight.data = decoder.model[0].weight.data.clone()
        self.model[0].bias.data = decoder.model[0].bias.data.clone()
        self.model[2].weight.data = decoder.model[2].weight.data.clone()
        self.model[2].bias.data = decoder.model[2].bias.data.clone()

    def forward(self, labels):
        z = self.latent_vector_generator.sample(labels)
        z = z.to(labels.device).float()  # Ensure z is on the correct device
        
        embedded_labels = self.embedding(labels)
        z = z.squeeze(1)
        input_vector = torch.cat([z, embedded_labels], dim=1)

        # 디버깅 출력 추가
        #print(f"Input vector shape: {input_vector.shape}")
        #print(f"Weight shape of the first layer: {self.model[0].weight.shape}")
        
        return self.model(z)
#%%

lr = 0.0001
num_epochs = 10
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


lr = 0.0005

num_classes = 10
latent_dim = 50


# LatentVectorGenerator 준비
latent_vector_generator = LatentVectorGenerator(num_classes=num_classes, latent_dim=latent_dim).to(device)

# LatentVectorGenerator 학습
train_latent_vector_generator(autoencoder[0], latent_vector_generator, data_loader, device)

discriminator = Discriminator(autoencoder[0], num_classes).to(device)
#generator = Generator(autoencoder[1], latent_dim, num_classes).to(device)

generator = Generator1(autoencoder[1], latent_vector_generator, num_classes = num_classes, latent_dim = latent_dim).to(device)

#discriminator = Discriminator(autoencoder[0].state_dict(), strict=False)
#generator = Generator(autoencoder[1].state_dict(), strict=False)


d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

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

        gen_labels = torch.LongTensor(np.random.randint(0, 9, batch_size)).to(device)
        fake_images = generator(gen_labels).to(device)
        outputs = discriminator(fake_images)
        g_loss = criterion1(outputs, torch.full((len(fake_images),), num_classes).to(device))
        
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








