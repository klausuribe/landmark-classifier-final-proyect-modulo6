"""
data.py
Módulo de manejo de datos: transforms, datasets y DataLoaders.

Funciones exportadas:
    - get_transforms(): retorna los transforms de train y val/test
    - get_dataloaders(): retorna los tres DataLoaders + lista de clases
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from pathlib import Path

# Constantes de normalización (estándar ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    """
    Retorna tupla (train_transform, val_test_transform).

    - train_transform: incluye data augmentation (flips, rotaciones, color jitter)
      para combatir overfitting en dataset chico.
    - val_test_transform: determinístico, solo resize + crop + normalización.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_test_transform


def get_dataloaders(data_dir="data", batch_size=32, val_split=0.15,
                    num_workers=2, seed=42):
    """
    Crea los DataLoaders de train, val y test.

    Args:
        data_dir: ruta a la carpeta que contiene train/ y test/
        batch_size: tamaño del batch
        val_split: fracción del train a reservar para validación (default 0.15)
        num_workers: workers paralelos (default 2, seguro para Windows)
        seed: semilla para reproducibilidad del split

    Returns:
        (train_loader, val_loader, test_loader, classes)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_transform, val_test_transform = get_transforms()

    # Dos "copias" del dataset de train: una con augmentation, otra sin
    full_train_aug = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    full_train_clean = datasets.ImageFolder(root=str(train_dir), transform=val_test_transform)

    # Split 85/15 reproducible
    n_total = len(full_train_aug)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    indices = random_split(range(n_total), [n_train, n_val], generator=generator)

    train_subset = Subset(full_train_aug, indices[0].indices)
    val_subset = Subset(full_train_clean, indices[1].indices)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=val_test_transform)

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_train_aug.classes
