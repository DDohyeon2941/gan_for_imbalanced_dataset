# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:05:01 2023

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
from models_classifier import ResNet
from models_binary_dataloader import make_concated_dataloader, load_test_loader
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix, roc_auc_score


"""
GAN 학습부터, 오버샘플링, 분류자학습까지 한번에 진행하는 모듈

"""



def train_wgan(data_dir = './data',
               minority_class=5,
               minority_size=500,
               param1=0.65,
               param2=0.2,
               batch_size=64,
               num_epochs=1000,
               z_dim=100,
               lr=0.0001):


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=False)

    #train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, drop_last=True)

    minority_data = [(data, label) for data, label in train_dataset if label in [minority_class,]][:minority_size]
    minority_loader = DataLoader(minority_data, batch_size= batch_size, shuffle=True, drop_last=True)


    # 하이퍼파라미터

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #device = torch.device('cpu')
    
    
    # 모델 및 옵티마이저 초기화
    generator = nn.DataParallel(Generator(z_dim))
    generator.to(device)
    discriminator1 = nn.DataParallel(Discriminator())
    discriminator1.to(device)
    
    
    
    discriminator2 = nn.DataParallel(Discriminator())
    discriminator2.load_state_dict(torch.load('discriminator_cifar10_car.pth'))
    discriminator2.to(device)
    
    
    g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
    d_optimizer1 = optim.RMSprop(discriminator1.parameters(), lr=lr)

    
    ###
    loss_dict = {'d_loss':[], 'g_loss':[],'grad':[]}

    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(minority_loader):
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

                gradients_min = grad(outputs=pred_hat_min, inputs=x_hat, grad_outputs=torch.ones(pred_hat_min.size()).to(device),
                                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                loss_dict['grad'].append(gradients_min.detach().cpu())

                gradient_penalty_min = cal_gradient(gradients_min)

                output_diff = torch.abs(d_loss_fake_min  - d_loss_fake_maj)
    
                d_loss = -torch.mean(d_loss_real) + (param1*torch.mean(d_loss_fake_min)) + (10*gradient_penalty_min) + ((1-param1)*torch.mean(d_loss_fake_maj)) + (param2 * output_diff.mean())
    

                d_loss.backward()
                d_optimizer1.step()
    
    
            #생성자 학습
            g_optimizer.zero_grad()
    
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = generator(z)
            output_min = discriminator1(fake_images)
            output_maj = discriminator2(fake_images)
    
            g_loss = -(param1*torch.mean(output_min)) - ((1-param1)*torch.mean(output_maj))
    
            g_loss.backward()
            g_optimizer.step()
    
    
            loss_dict['d_loss'].append(d_loss.item())
            loss_dict['g_loss'].append(g_loss.item())
            print(f"Epoch [{epoch + 1}/{num_epochs}] D Loss: {d_loss.item()} G Loss: {g_loss.item()}")

        if epoch % 10 == 0:
            show_generated_imgs(generator, z_dim, device)

    return generator, loss_dict

def generate_images(generator, z_dim= 100, num_images=1000):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 이미지 생성 및 저장
    generator.eval() # 생성자를 평가 모드로 설정
    with torch.no_grad(): # 그라디언트 계산을 하지 않음
        z = torch.randn(num_images, z_dim, 1, 1, device=device)
        generated_images = generator(z).detach().cpu()
    return generated_images


#%%
if __name__ == '__main__':
    trained_generator, loss_dict1 = train_wgan(param1=0.95, param2=0.1, num_epochs=300, z_dim=128, minority_class=9, minority_size = 300, lr=0.0003)

    #sum([(loss_dict1['grad'][xx]>1).sum() for xx in range(len(loss_dict1['grad']))])
    generated_images = generate_images(trained_generator, z_dim=128, num_images=5500)
    train_loader = make_concated_dataloader(generated_images, batch_size=128, majority_class=1, minority_class=9)

    # 이 모델을 GPU로 이동
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 모델 인스턴스 생성
    resnet_model = nn.DataParallel(ResNet(num_classes=2))  # 2개의 클래스가 있다고 가정
    resnet_model.to(device)


    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)

    # 학습 과정
    num_epochs = 10  # 학습 에포크 수

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 데이터를 GPU로 이동
            inputs, labels = inputs.to(device), labels.to(device)

            # 옵티마이저의 그래디언트를 0으로 설정
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = resnet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 출력
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Finished Training')

    print("================test==========================================")

    test_loader = load_test_loader(batch_size=128, majority_class=1, minority_class=9)
    resnet_model.eval()  # 모델을 평가 모드로 설정합니다.

    all_labels = []
    all_preds = []
    all_probs = []

    # 테스트 데이터에 대한 예측 수행
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = resnet_model(inputs)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 이진 분류의 경우, 클래스 1에 대한 확률을 선택합니다.


            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # 성능 측정
    print(classification_report(all_labels, all_preds))
    print(f'Accuracy: {accuracy_score(all_labels, all_preds):.4f}')
    print(f"AUROC: {roc_auc_score(all_labels, all_probs):.4f}")
    print(confusion_matrix(all_labels, all_preds))
    
