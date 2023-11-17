# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:03:11 2023

@author: dohyeon
"""
import torch.nn as nn
import torchvision.models as models

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
