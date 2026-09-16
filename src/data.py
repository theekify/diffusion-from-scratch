import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def normalize_to_unit_range(x):
    return (x * 2) - 1


def get_fashion_mnist_dataloader(batch_size: int = 128, image_size: int = 28, train: bool = True, num_workers: int = 0):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(normalize_to_unit_range),
    ])

    dataset = datasets.FashionMNIST(
        root="./data", train=train, download=True, transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=True)