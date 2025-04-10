import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from collections import Counter

class DataPreprocessor:
    def __init__(self, root_dir='./data', batch_size=16, oversampling_factor=1.0):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.oversampling_factor = oversampling_factor
        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        train_dataset = datasets.ImageFolder(f'{self.root_dir}/train', transform=self.train_transform)
        val_dataset = datasets.ImageFolder(f'{self.root_dir}/val', transform=self.val_transform)
        test_dataset = datasets.ImageFolder(f'{self.root_dir}/test', transform=self.val_transform)

        oversampled_train_dataset = self.oversample_dataset(train_dataset)
        train_loader = DataLoader(oversampled_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader

    def oversample_dataset(self, dataset):
        label_counts = Counter(dataset.targets)
        target_count = max(label_counts.values())
        oversampled_datasets = []
        for class_idx in range(len(label_counts)):
            class_samples = [sample for sample in dataset if sample[1] == class_idx]
            class_count = len(class_samples)
            if class_count < target_count:
                num_to_add = int((target_count - class_count) * self.oversampling_factor)
                augmented_samples = []
                for _ in range(num_to_add):
                    sample = class_samples[np.random.randint(class_count)]
                    augmented_image = self.augmentation_transforms(sample[0])
                    augmented_samples.append((augmented_image, class_idx))
                oversampled_datasets.append(ConcatDataset([dataset, augmented_samples]))
            else:
                oversampled_datasets.append(dataset)
        return ConcatDataset(oversampled_datasets)

    def augmentation_transforms(self, image):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), antialias=True),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])(image)