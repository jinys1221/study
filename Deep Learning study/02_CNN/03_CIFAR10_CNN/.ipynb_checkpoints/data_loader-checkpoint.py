# CIFAR10_Classfication/data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=64, transform_train=None, transform_test=None):

    # Train 데이터 전처리 (Augmentation 포함)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))    
    ])

    # Test 데이터 전처리 (Augmentation 없이 정규화만)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # 데이터셋 다운로드 및 로드
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    # DataLoader 구성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

