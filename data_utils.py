# data_utils.py
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from collections import Counter
import os
from tqdm import tqdm
import numpy as np
import random

def load_data(transform):
    # 加载数据集
    train_dataset = datasets.ImageFolder(configs.train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(configs.val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(configs.test_dir, transform=transform)
    return train_dataset, val_dataset, test_dataset

def get_class_counts(dataset):
    # 计算每个类别的样本数量
    class_counts = Counter(dataset.targets)
    return class_counts

def oversample_dataset(dataset, class_counts, target_count, oversampling_factor=1.0):
    # 动态过采样
    oversampled_datasets = []
    for class_idx in range(len(class_counts)):
        class_samples = [sample for sample in dataset if sample[1] == class_idx]
        class_count = class_counts[class_idx]
        if class_count < target_count:
            num_to_add = int((target_count - class_count) * oversampling_factor)
            augmented_samples = []
            for _ in tqdm(range(num_to_add), desc=f"Augmenting class {class_idx}"):
                sample = class_samples[np.random.randint(class_count)]
                augmented_image = augmentation_transforms(sample[0])
                augmented_samples.append((augmented_image, class_idx))
            oversampled_datasets.append(ConcatDataset([dataset, augmented_samples]))
        else:
            oversampled_datasets.append(dataset)
    return ConcatDataset(oversampled_datasets)

def create_data_loaders(gpu_mode=False):
    # 定义数据增强
    augmentation_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), antialias=True),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 定义变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_data(train_transform)
    
    # 计算类别分布
    label_counts = get_class_counts(train_dataset)
    target_count = max(label_counts.values())
    
    # 动态过采样
    oversampled_train_dataset = oversample_dataset(train_dataset, label_counts, target_count, oversampling_factor=1.0)
    train_loader = DataLoader(oversampled_train_dataset, batch_size=configs.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader