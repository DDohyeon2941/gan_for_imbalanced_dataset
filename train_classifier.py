# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:23:47 2023

@author: dohyeon
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
import torchvision.models as models

class ImbalancedBinaryCIFAR10Dataset(Dataset):
    def __init__(self, cifar10_dataset, transform=None, majority_class=4, minority_class=5, minority_size=200):
        self.cifar10_dataset = cifar10_dataset
        self.transform = transform
        self.data = []
        self.labels = []
        for data, label in self.cifar10_dataset:
            if label == minority_class and len([lbl for lbl in self.labels if lbl == 1]) <= minority_size:
                self.data.append(data)
                self.labels.append(1)  # 라벨 4를 0으로 변환
            elif label == majority_class:
                self.data.append(data)
                self.labels.append(0)  # 라벨 5를 1로 변환
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(label, dtype=torch.long)

class BinaryCIFAR10Dataset(Dataset):
    def __init__(self, cifar10_dataset, transform=None, majority_class=4, minority_class=5):
        self.cifar10_dataset = cifar10_dataset
        self.transform = transform
        self.data = []
        self.labels = []
        for data, label in self.cifar10_dataset:
            if label == minority_class:
                self.data.append(data)
                self.labels.append(1)  # 라벨 4를 0으로 변환
            elif label == majority_class:
                self.data.append(data)
                self.labels.append(0)  # 라벨 5를 1로 변환
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(label, dtype=torch.long)



# ResNet 모델 정의
class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        # ResNet50을 불러옵니다. pretrained=True로 하면 사전 학습된 가중치를 사용합니다.
        self.model = models.resnet50(pretrained=False)
        
        # ResNet의 마지막 선형 레이어를 교체합니다. ResNet50의 경우 2048개의 특성이 있습니다.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
#%%

# cifar10 데이터셋 로딩
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)


# 불균형 이진 분류용 데이터셋 생성
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


imbalanced_binary_dataset = ImbalancedBinaryCIFAR10Dataset(train_dataset,
                                                         transform=transform,
                                                         minority_class=5,
                                                         majority_class=3,
                                                         minority_size=500)


#%%

# 저장된 텐서 불러오기
loaded_images = torch.load('saved_tensor.pt')

# labels가 없는 경우, 가짜 레이블을 생성할 수 있습니다.
# 여기서는 1으로 가정했습니다.
fake_labels = torch.ones(loaded_images.size(0), dtype=torch.long)

# TensorDataset 생성
generated_dataset = TensorDataset(loaded_images, fake_labels)

combined_dataset = ConcatDataset([imbalanced_binary_dataset, generated_dataset])

#%%

def get_labels_from_concat_dataset(concat_dataset):
    labels = []
    for dataset in concat_dataset.datasets:
        # 데이터셋의 라벨을 추출하는 방법은 데이터셋의 형태에 따라 다를 수 있습니다.
        labels += [label for _, label in dataset]
    return np.array(labels)

# ConcatDataset 객체 생성 후
all_labels = get_labels_from_concat_dataset(combined_dataset)

#%%
import numpy as np

# 라벨에 따라 각 샘플의 가중치를 계산합니다.
class_sample_counts = [len(np.where(all_labels == t)[0]) for t in np.unique(all_labels)]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
samples_weights = weights[all_labels]

# WeightedRandomSampler를 생성합니다.
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

# DataLoader에 sampler를 전달합니다.
combined_loader = DataLoader(combined_dataset, batch_size=64, sampler=sampler)


#%%

# 모델 인스턴스 생성
resnet_model = ResNet(num_classes=2)  # 2개의 클래스가 있다고 가정

# 이 모델을 GPU로 이동
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet_model = resnet_model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)

# 학습 과정
num_epochs = 10  # 학습 에포크 수

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in combined_loader:
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
    
    epoch_loss = running_loss / len(combined_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

print('Finished Training')


#%%


test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)


test_binary_dataset = BinaryCIFAR10Dataset(test_dataset,
                                           transform=transform,
                                           minority_class=5,
                                           majority_class=3)

test_loader = DataLoader(test_binary_dataset, batch_size=64, shuffle=True)
#%%

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix, roc_auc_score

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