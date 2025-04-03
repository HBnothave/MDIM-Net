import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import ConcatDataset

def prepare_data(oversampling_factor=1.0):
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('./data/NCT/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('./data/NCT/val', transform=val_transform)
    test_dataset = datasets.ImageFolder('./data/NCT/test', transform=val_transform)

    label_counts = Counter(train_dataset.targets)
    target_count = max(label_counts.values())
    oversampled_train_dataset = oversample_dataset(train_dataset, label_counts, target_count, oversampling_factor)

    return (
        data.DataLoader(oversampled_train_dataset, batch_size=16, shuffle=True, num_workers=2),
        data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2),
        data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    )

def oversample_dataset(dataset, label_counts, target_count, oversampling_factor=1.0):
    oversampled_datasets = []
    for class_idx in range(len(label_counts)):
        class_samples = [sample for sample in dataset if sample[1] == class_idx]
        class_count = len(class_samples)
        if class_count < target_count:
            num_to_add = int((target_count - class_count) * oversampling_factor)
            augmented_samples = []
            for _ in range(num_to_add):
                sample = class_samples[np.random.randint(class_count)]
                augmented_image = augmentation_transforms(sample[0])
                augmented_samples.append((augmented_image, class_idx))
            oversampled_datasets.append(ConcatDataset([dataset, augmented_samples]))
        else:
            oversampled_datasets.append(dataset)
    return ConcatDataset(oversampled_datasets)

def augmentation_transforms(image):
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), antialias=True),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])(image)