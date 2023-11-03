# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:38 2023

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
minority_data = [(data, label) for data, label in train_dataset if label in [0,]][:200]
minority_loader = DataLoader(minority_data, batch_size= 32, shuffle=True, drop_last=True)


class Generator(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.embedding = nn.Embedding(num_classes, z_dim)
        self.model = nn.Sequential(
            nn.Linear(z_dim*2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
            )
    def forward(self, z, labels):
        #z = z.view(z.size(0), 100)
        #c = self.label_emb(label)
        labels_embedding = self.embedding(labels)
        x = torch.cat([z,labels_embedding], 1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        #self.label_emb = nn.Embedding(5,2)
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.disc_linear = nn.Linear(256, 1)
        self.aux_linear = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        disc_logits = self.disc_linear(x)
        aux_logits = self.aux_linear(x)
        return self.sigmoid(disc_logits), self.softmax(aux_logits)




#%%
z_dim = 100
num_classes = 2
num_epochs = 100
batch_size = 32
lr = 0.0005

#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(z_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()


d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

#d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
#g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


for epoch in range(num_epochs):
    for i, (real_data, labels) in enumerate(minority_loader):
        #real_data = real_data.view(real_data.size(0),-1).to(device)
        real_data = real_data.to(device)
        labels = labels.to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 판별자 학습
        d_optimizer.zero_grad()

        validity_real, pred_label_real = discriminator(real_data)
        d_real_loss = (adversarial_loss(validity_real.to(device), real_labels) + auxiliary_loss(pred_label_real.to(device), labels)) / 2

        z = torch.randn(batch_size, z_dim).to(device)
        gen_labels = torch.randint(0, 1, (batch_size,)).to(device)
        fake_images = generator(z, gen_labels)

        validity_fake, pred_label_fake = discriminator(fake_images.detach())
        d_fake_loss = (adversarial_loss(validity_fake.to(device), fake_labels) + auxiliary_loss(pred_label_fake.to(device), gen_labels)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        d_optimizer.step()

        #생성자 학습

        g_optimizer.zero_grad()

        validity, pred_label = discriminator(fake_images.to(device))
        g_loss = adversarial_loss(validity.to(device), real_labels) + auxiliary_loss(pred_label.to(device), gen_labels)

        g_loss.backward()
        g_optimizer.step()

    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(minority_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")


#%%

# 학습된 생성자를 이용해서 가짜 샘플 생성
import matplotlib.pyplot as plt

with torch.no_grad():
    z = torch.randn(batch_size, z_dim).to(device)  # 64개의 랜덤 벡터 생성
    gen_labels = torch.LongTensor(np.random.randint(0, 1, z.shape[0])).to(device)
    fake_images = generator(z, gen_labels).detach().cpu()
    fake_images = fake_images.reshape(fake_images.shape[0], 1, 28, 28)

# 이미지 출력
fig, axes = plt.subplots(1, 8, figsize=(20, 2))
for i in range(8):
    axes[i].imshow(torchvision.utils.make_grid(fake_images[i], normalize=True).permute(1, 2, 0))
    axes[i].axis('off')
plt.show()
#print(gen_labels[:8])





















