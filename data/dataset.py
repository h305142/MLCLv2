# data/dataset.py
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from torch.utils.data import DataLoader


class TwoCropsTransform:
    """Generate two different augmented versions of the same image"""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


# Unified get_dataset function
def get_dataset(dataset_name):
    """
    Return dataset based on dataset name (does not return loader)
    
    Args:
        dataset_name: str, 'cifar10', 'cifar100', or 'stl10'
    
    Returns:
        train_dataset: Dataset object
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        train_dataset = CIFAR10(
            root='./data',
            train=True,
            transform=TwoCropsTransform(augmentation),
            download=True
        )
    
    elif dataset_name == 'cifar100':
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
        train_dataset = CIFAR100(
            root='./data',
            train=True,
            transform=TwoCropsTransform(augmentation),
            download=True
        )
    
    elif dataset_name == 'stl10':
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = STL10(
            root='./data',
            split='unlabeled',
            transform=TwoCropsTransform(augmentation),
            download=True
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                       f"Supported: 'cifar10', 'cifar100', 'stl10'")
    
    return train_dataset


# Main function with return_dataset parameter support
def get_dataloader(dataset_name, batch_size=256, num_workers=4, return_dataset=False):
    """
    Return dataloader or dataset based on dataset name
    
    Args:
        dataset_name: str, 'cifar10', 'cifar100', or 'stl10'
        batch_size: int
        num_workers: int
        return_dataset: bool, if True, return dataset; otherwise return dataloader
    
    Returns:
        train_dataset (if return_dataset=True) or train_loader (if return_dataset=False)
    """
    train_dataset = get_dataset(dataset_name)
    
    # DDP mode requires dataset, single GPU mode requires loader
    if return_dataset:
        return train_dataset
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        print(f"{dataset_name.upper()} dataset loaded: {len(train_dataset)} training images")
        
        return train_loader


# Preserve original standalone functions for backward compatibility
def get_cifar10_loaders(batch_size=256, num_workers=4):
    """CIFAR-10 dataloader (backward compatible)"""
    return get_dataloader('cifar10', batch_size, num_workers)


def get_cifar100_loaders(batch_size=256, num_workers=4):
    """CIFAR-100 dataloader (backward compatible)"""
    return get_dataloader('cifar100', batch_size, num_workers)


def get_stl10_loaders(batch_size=256, num_workers=4):
    """STL-10 dataloader (backward compatible)"""
    return get_dataloader('stl10', batch_size, num_workers)