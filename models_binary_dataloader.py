# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:23:47 2023

@author: dohyeon
"""

import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset, WeightedRandomSampler

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

class ImbalancedBinaryCIFAR10Dataset(BinaryCIFAR10Dataset):
    def __init__(self,
                 cifar10_dataset,
                 transform=None,
                 majority_class=4,
                 minority_class=5,
                 minority_size=200):
        super().__init__(cifar10_dataset=cifar10_dataset,
                         transform=transform,
                         majority_class = majority_class,
                         minority_class = minority_class)
        self.data = []
        self.labels = []
        for data, label in self.cifar10_dataset:
            if label == minority_class and len([lbl for lbl in self.labels if lbl == 1]) <= minority_size:
                self.data.append(data)
                self.labels.append(1)  # 라벨 4를 0으로 변환
            elif label == majority_class:
                self.data.append(data)
                self.labels.append(0)  # 라벨 5를 1로 변환
                
def get_labels_from_concat_dataset(concat_dataset):
    labels = []
    for dataset in concat_dataset.datasets:
        # 데이터셋의 라벨을 추출하는 방법은 데이터셋의 형태에 따라 다를 수 있습니다.
        labels += [label for _, label in dataset]
    return np.array(labels)

def make_sampler_concated_dataloader(generated_images,
                             data_dir='./data',
                             minority_class=5,
                             majority_class=3,
                             minority_size=500,
                             batch_size=64):
    # cifar10 데이터셋 로딩
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
    
    
    # 불균형 이진 분류용 데이터셋 생성
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    
    imbalanced_binary_dataset = ImbalancedBinaryCIFAR10Dataset(train_dataset,
                                                             transform=transform,
                                                             minority_class=minority_class,
                                                             majority_class=majority_class,
                                                             minority_size=minority_size)
    fake_labels = torch.ones(generated_images.size(0), dtype=torch.long)
    # TensorDataset 생성
    generated_dataset = TensorDataset(generated_images, fake_labels)
    combined_dataset = ConcatDataset([imbalanced_binary_dataset, generated_dataset])

    # ConcatDataset 객체 생성 후
    all_labels = get_labels_from_concat_dataset(combined_dataset)


    # 라벨에 따라 각 샘플의 가중치를 계산합니다.
    class_sample_counts = [len(np.where(all_labels == t)[0]) for t in np.unique(all_labels)]
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[all_labels]
    
    # WeightedRandomSampler를 생성합니다.
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    
    # DataLoader에 sampler를 전달합니다.
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)

    return combined_loader


def make_concated_dataloader(generated_images,
                             data_dir='./data',
                             minority_class=5,
                             majority_class=3,
                             minority_size=500,
                             batch_size=64):
    # cifar10 데이터셋 로딩
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
    
    
    # 불균형 이진 분류용 데이터셋 생성
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    
    imbalanced_binary_dataset = ImbalancedBinaryCIFAR10Dataset(train_dataset,
                                                             transform=transform,
                                                             minority_class=minority_class,
                                                             majority_class=majority_class,
                                                             minority_size=minority_size)
    fake_labels = torch.ones(generated_images.size(0), dtype=torch.long)
    # TensorDataset 생성
    generated_dataset = TensorDataset(generated_images, fake_labels)
    combined_dataset = ConcatDataset([imbalanced_binary_dataset, generated_dataset])


    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return combined_loader







def load_test_loader(data_dir='./data', minority_class=5, majority_class=3, batch_size=64):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False)


    test_binary_dataset = BinaryCIFAR10Dataset(test_dataset,
                                               transform=transform,
                                               minority_class=minority_class,
                                               majority_class=majority_class)

    test_loader = DataLoader(test_binary_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
