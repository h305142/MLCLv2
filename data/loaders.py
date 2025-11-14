# data/loaders.py
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from .transforms import (
    get_contrastive_transforms,
    get_ptr_transform,
    get_nu_prior_augmentation,
    get_val_transforms
)
from .datasets import FourViewDataset, ColoredMNIST, ColoredMNISTFourView
from config.base_config import ExperimentConfig


# ========================================
# Custom Collate Function
# ========================================

def custom_collate_fn(batch):
    """
    Custom collate function to handle four-view data
    
    Input: batch = [
        ((im_q_1, im_k_1, ptr_1, nu_1), target_1, target_nu_1, flag_1),
        ((im_q_2, im_k_2, ptr_2, nu_2), target_2, target_nu_2, flag_2),
        ...
    ]
    
    Output:
        - (im_q_batch, im_k_batch, ptr_batch, nu_batch): Four tensors
        - target_batch: tensor (B,)
        - target_nu_batch: tensor (B,) or dict (if Mixup is used)
        - flag_batch: tensor (B,) bool
    """
    # Extract each field separately
    views_list = []
    target_list = []
    target_nu_list = []
    flag_list = []
    
    for item in batch:
        (im_q, im_k, ptr, nu), target, target_nu, flag = item
        views_list.append((im_q, im_k, ptr, nu))
        target_list.append(target)
        target_nu_list.append(target_nu)
        flag_list.append(flag)
    
    # 1. Process four views
    im_q_batch = torch.stack([v[0] for v in views_list], dim=0)
    im_k_batch = torch.stack([v[1] for v in views_list], dim=0)
    ptr_batch = torch.stack([v[2] for v in views_list], dim=0)
    nu_batch = torch.stack([v[3] for v in views_list], dim=0)
    
    # 2. Hard labels
    target_batch = torch.tensor(target_list, dtype=torch.long)
    
    # 3. Flag
    flag_batch = torch.tensor(flag_list, dtype=torch.bool)
    
    # 4. target_nu (may contain soft labels)
    has_mixup = any(isinstance(t, tuple) for t in target_nu_list)
    
    if has_mixup:
        batch_size = len(target_nu_list)
        lam_list = []
        target1_list = []
        target2_list = []
        
        for target_nu in target_nu_list:
            if isinstance(target_nu, tuple):
                lam, t1, t2 = target_nu
                lam_list.append(lam)
                target1_list.append(t1)
                target2_list.append(t2)
            else:
                lam_list.append(1.0)
                target1_list.append(target_nu)
                target2_list.append(target_nu)
        
        target_nu_batch = {
            'lam': torch.tensor(lam_list, dtype=torch.float32),
            'target1': torch.tensor(target1_list, dtype=torch.long),
            'target2': torch.tensor(target2_list, dtype=torch.long)
        }
    else:
        target_nu_batch = torch.tensor(target_nu_list, dtype=torch.long)
    
    return (im_q_batch, im_k_batch, ptr_batch, nu_batch), target_batch, target_nu_batch, flag_batch


# ========================================
# Data Loader Creation Functions
# ========================================

def create_data_loaders(config: ExperimentConfig):
    """Create data loaders based on configuration"""
    if config.dataset == 'cifar10':
        return create_cifar10_loaders(config)
    elif config.dataset == 'cifar100':
        return create_cifar100_loaders(config)
    elif config.dataset == 'colored_mnist':
        return create_colored_mnist_loaders(config)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")


def create_cifar10_loaders(config: ExperimentConfig):
    """Create CIFAR-10 data loaders"""
    
    # 1. Get all augmentations
    contrastive_q, contrastive_k = get_contrastive_transforms('cifar')
    ptr_transform = get_ptr_transform('cifar')
    nu_transform = get_nu_prior_augmentation(
        'cifar', 
        strategy=config.loss.nu_augmentation_strategy
    )
    val_transform = get_val_transforms('cifar')
    
    # 2. Create base datasets
    base_train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True,
        transform=None
    )
    
    # 3. Wrap as four-view dataset
    train_dataset = FourViewDataset(
        base_train_dataset,
        contrastive_transform_q=contrastive_q,
        contrastive_transform_k=contrastive_k,
        ptr_transform=ptr_transform,
        nu_transform=nu_transform,
        mixup_nu=config.loss.mixup_nu,
        mixup_alpha=config.loss.mixup_alpha,
        mixup_ratio=config.loss.mixup_ratio,
        noise_nu=config.loss.noise_nu,
        noise_std=config.loss.noise_std,
        noise_ratio=config.loss.noise_ratio
    )
    
    val_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    print(f"✓ CIFAR-10 loaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  ν augmentation strategy: {config.loss.nu_augmentation_strategy}")
    
    return train_loader, val_loader


def create_cifar100_loaders(config: ExperimentConfig):
    """Create CIFAR-100 data loaders (same as CIFAR-10)"""
    
    contrastive_q, contrastive_k = get_contrastive_transforms('cifar')
    ptr_transform = get_ptr_transform('cifar')
    nu_transform = get_nu_prior_augmentation(
        'cifar', 
        strategy=config.loss.nu_augmentation_strategy
    )
    val_transform = get_val_transforms('cifar')
    
    base_train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True,
        transform=None
    )
    
    train_dataset = FourViewDataset(
        base_train_dataset,
        contrastive_transform_q=contrastive_q,
        contrastive_transform_k=contrastive_k,
        ptr_transform=ptr_transform,
        nu_transform=nu_transform,
        mixup_nu=config.loss.mixup_nu,
        mixup_alpha=config.loss.mixup_alpha,
        mixup_ratio=config.loss.mixup_ratio,
        noise_nu=config.loss.noise_nu,
        noise_std=config.loss.noise_std,
        noise_ratio=config.loss.noise_ratio
    )
    
    val_dataset = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    print(f"✓ CIFAR-100 loaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  ν augmentation strategy: {config.loss.nu_augmentation_strategy}")
    
    return train_loader, val_loader


def create_colored_mnist_loaders(config: ExperimentConfig):
    """Create Colored MNIST data loaders"""
    
    contrastive_q, contrastive_k = get_contrastive_transforms('mnist')
    ptr_transform = get_ptr_transform('mnist')
    nu_transform = get_nu_prior_augmentation(
        'mnist', 
        strategy=config.loss.nu_augmentation_strategy
    )
    val_transform = get_val_transforms('mnist')
    
    base_train_dataset = ColoredMNIST(
        root='./data',
        train=True,
        download=True,
        transform=None,
        color_correlation=0.9
    )
    
    train_dataset = ColoredMNISTFourView(
        base_train_dataset,
        contrastive_transform_q=contrastive_q,
        contrastive_transform_k=contrastive_k,
        ptr_transform=ptr_transform,
        nu_transform=nu_transform,
        color_flip_ratio=config.loss.color_flip_ratio
    )
    
    val_dataset = ColoredMNIST(
        root='./data',
        train=False,
        download=True,
        transform=val_transform,
        color_correlation=0.1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    print(f"✓ Colored MNIST loaders created:")
    print(f"  Train: 90% color correlation")
    print(f"  Val: 10% color correlation")
    print(f"  ν augmentation strategy: {config.loss.nu_augmentation_strategy}")
    
    return train_loader, val_loader