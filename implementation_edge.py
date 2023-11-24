# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:53:13 2023

@author: dohyeon
"""
from models_binary_dataloader import  load_test_loader
import matplotlib.pyplot as plt
#import cv2
#%%
def visualize_tensor(tensor_image):
    # 텐서 이미지가 'C x H x W' 형식인 경우, 'H x W x C'로 변환
    if tensor_image.shape[0] < tensor_image.shape[2]:
        tensor_image = tensor_image.transpose(0, 1).transpose(1, 2)

    tensor_image = tensor_image.numpy()  # 텐서를 넘파이 배열로 변환
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    plt.imshow(tensor_image)
    plt.axis('off')  # 축 표시 제거
    plt.show()

# 이미지 시각화 실행
visualize_tensor(inputs.detach()[0])

#%%

test_loader = load_test_loader(batch_size=128, majority_class=1, minority_class=9)
for inputs, labels in test_loader: break

#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_tensor_with_edges(tensor_image):
    # 이미지를 'C x H x W'에서 'H x W x C'로 변환
    if tensor_image.shape[0] < tensor_image.shape[2]:
        tensor_image = tensor_image.transpose(0, 1).transpose(1, 2)

    # 이미지 데이터를 [0, 1] 범위로 정규화
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    # 이미지를 회색조로 변환
    tensor_image_gray = np.dot(tensor_image[..., :3], [0.2989, 0.5870, 0.1140])
    tensor_image_gray = torch.tensor(tensor_image_gray)

    # Sobel 필터를 Float 자료형으로 정의
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # 이미지 텐서의 자료형을 Float으로 변환
    tensor_image_gray = tensor_image_gray.type(torch.float32)
    
    # 에지 검출을 위한 컨볼루션 수행
    edge_x = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    edge_y = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0).squeeze(0)

    # 원본 이미지와 에지 이미지 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(tensor_image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(edges, cmap='gray')
    ax2.set_title("Edge Detection")
    ax2.axis('off')

    plt.show()

# 이미지 시각화 실행
visualize_tensor_with_edges(inputs.detach()[20])
#%%
def visualize_edges_only(tensor_image):
    # 이미지를 'C x H x W'에서 'H x W x C'로 변환
    if tensor_image.shape[0] < tensor_image.shape[2]:
        tensor_image = tensor_image.transpose(0, 1).transpose(1, 2)

    # 이미지 데이터를 [0, 1] 범위로 정규화
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    # 이미지를 회색조로 변환하고 자료형을 float32로 변경
    tensor_image_gray = np.dot(tensor_image[..., :3], [0.2989, 0.5870, 0.1140])
    tensor_image_gray = torch.tensor(tensor_image_gray, dtype=torch.float32)
    
    # Sobel 필터를 정의 (이미 float32로 정의되었음)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # 에지 검출을 위한 컨볼루션 수행
    edge_x = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    edge_y = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0).squeeze(0)
    # 에지 이외의 부분을 흰색으로 만듦
    edges_only = np.ones(tensor_image.shape)  # 전체를 흰색으로 초기화
    edges_only[edges.numpy() > 0] = 0  # 에지 부분만 검은색으로 변경

    # 에지만 표시된 이미지 시각화
    plt.imshow(edges_only, cmap='gray')
    plt.axis('off')
    plt.show()

# 이미지 시각화 실행
visualize_edges_only(inputs.detach()[0])
#%%

def visualize_edges_only(tensor_image):
    # 이미지를 'C x H x W'에서 'H x W x C'로 변환
    if tensor_image.shape[0] < tensor_image.shape[2]:
        tensor_image = tensor_image.transpose(0, 1).transpose(1, 2)

    # 이미지 데이터를 [0, 1] 범위로 정규화
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    # 이미지를 회색조로 변환하고 자료형을 float32로 변경
    tensor_image_gray = np.dot(tensor_image[..., :3], [0.2989, 0.5870, 0.1140])
    tensor_image_gray = torch.tensor(tensor_image_gray, dtype=torch.float32)

    # Sobel 필터를 정의
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    # 에지 검출을 위한 컨볼루션 수행
    edge_x = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    edge_y = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0).squeeze(0)

    # 에지 강도에 대한 임계값 설정
    threshold = edges.max() * 0.32 # 임계값을 최대 에지 강도의 30%로 설정
    edges_binary = edges > threshold

    # 에지만 검은색으로, 나머지는 흰색으로 설정
    edges_only = np.ones(tensor_image.shape)
    edges_only[edges_binary.numpy()] = 0

    # 에지만 표시된 이미지 시각화
    plt.imshow(edges_only, cmap='gray')
    plt.axis('off')
    plt.show()

# 이미지 시각화 실행
visualize_edges_only(inputs.detach().cpu()[0])
#%%

def visualize_tensor_with_edges(tensor_image):
    # 이미지를 'C x H x W'에서 'H x W x C'로 변환
    if tensor_image.shape[0] < tensor_image.shape[2]:
        tensor_image = tensor_image.transpose(0, 1).transpose(1, 2)

    # 이미지 데이터를 [0, 1] 범위로 정규화
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    # 이미지를 회색조로 변환
    tensor_image_gray = np.dot(tensor_image[..., :3], [0.2989, 0.5870, 0.1140])
    tensor_image_gray = torch.tensor(tensor_image_gray)

    # Sobel 필터를 Float 자료형으로 정의
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # 이미지 텐서의 자료형을 Float으로 변환
    tensor_image_gray = tensor_image_gray.type(torch.float32)
    
    # 에지 검출을 위한 컨볼루션 수행
    edge_x = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    edge_y = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0).squeeze(0)

    # 원본 이미지와 에지 이미지 시각화
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 6))

    ax2.imshow(edges, cmap='gray')
    ax2.set_title("Edge Detection")
    ax2.axis('off')

    plt.show()

# 이미지 시각화 실행
visualize_tensor_with_edges(inputs.detach()[1])


#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

def increase_resolution(tensor_image, scale_factor=2):
    # 이미지를 'C x H x W'에서 'H x W x C'로 변환
    if tensor_image.shape[0] < tensor_image.shape[2]:
        tensor_image = tensor_image.transpose(0, 1).transpose(1, 2)

    # 이미지 데이터를 [0, 1] 범위로 정규화
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    # PIL 이미지로 변환
    pil_image = Image.fromarray((tensor_image.numpy() * 255).astype(np.uint8))

    # 이미지 크기 증가
    original_size = pil_image.size
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    high_res_image = pil_image.resize(new_size, Image.BICUBIC)

    return np.array(high_res_image) / 255.0

def visualize_tensor_with_edges(tensor_image, scale_factor=2):
    # 이미지 해상도 증가
    high_res_image = increase_resolution(tensor_image, scale_factor)

    # 회색조로 변환
    tensor_image_gray = np.dot(high_res_image[..., :3], [0.2989, 0.5870, 0.1140])
    tensor_image_gray = torch.tensor(tensor_image_gray, dtype=torch.float32)

    # Sobel 필터 정의
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    # 에지 검출을 위한 컨볼루션 수행
    edge_x = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    edge_y = torch.nn.functional.conv2d(tensor_image_gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0).squeeze(0)

    # 원본 이미지와 에지 이미지 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(high_res_image)
    ax1.set_title("High Resolution Image")
    ax1.axis('off')

    ax2.imshow(edges, cmap='gray')
    ax2.set_title("Edge Detection")
    ax2.axis('off')

    plt.show()
#%%
# 이미지 시각화 실행
visualize_tensor_with_edges(inputs.detach()[120])
