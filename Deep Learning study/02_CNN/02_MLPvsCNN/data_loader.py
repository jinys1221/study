# MLPvsCNN/data_loader.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=64):

    # 데이터 전처리 정의 (ToTensor만 사용)
    transform = transforms.ToTensor()

    # 데이터셋 다운로드 및 로드
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    # DataLoader 구성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader