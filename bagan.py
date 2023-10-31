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
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):

        return x


class Discriminator(nn.Module):
    def __init__(self, encoder : nn.Module):
        super(Discriminator, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder(x)

        return x

class Generator(nn.Module):
    def __init__(self, decoder: nn.Module):
        super(Generator, self).__init__()
        self.decoder = decoder

    def forward(self, z):
        x = self.decoder(z)
        return x
#

lr = 0.0001
num_epochs = 10

# Load and preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


autoencoder = nn.Sequential(Encoder(), Decoder())
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(data_loader):
        ae_optimizer.zero_grad()
        recon = autoencoder(imgs)
        ae_loss = nn.MSELoss(recon, imgs)
        ae_loss.backward()
        ae_optimizer.step()

#Initialize the GAN with the trained autoencoder weights

discriminator = Discriminator(autoencoder[0])
generator = Generator(autoencoder[1])

