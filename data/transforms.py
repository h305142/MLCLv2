# data/transforms.py
from torchvision import transforms
from PIL import ImageFilter
import random


class GaussianBlur:
    """Gaussian blur augmentation (for MoCo)"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# ========================================
# 1. Contrastive Learning Augmentation (for im_q, im_k)
# ========================================

def get_contrastive_transforms(dataset_type='cifar'):
    """
    Get two view augmentations for contrastive learning
    
    Design: Standard MoCo-style augmentation
    - im_q: Standard augmentation
    - im_k: Standard augmentation + Gaussian blur
    """
    if dataset_type == 'cifar':
        # Contrastive learning augmentation for CIFAR-10/100
        base_transform = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        
        # Query view (im_q): Standard augmentation
        transform_q = transforms.Compose([
            *base_transform,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # Key view (im_k): Standard augmentation + Gaussian blur
        transform_k = transforms.Compose([
            *base_transform,
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        return transform_q, transform_k
    
    elif dataset_type == 'mnist':
        # Contrastive learning augmentation for MNIST
        base_transform = [
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
        ]
        
        transform_q = transforms.Compose([
            *base_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        transform_k = transforms.Compose([
            *base_transform,
            transforms.RandomApply([GaussianBlur([0.1, 1.0])], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform_q, transform_k
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# ========================================
# 2. DRO Distribution Augmentation (for ptr, nu)
# ========================================

def get_ptr_transform(dataset_type='cifar'):
    """
    Augmentation for P_tr: Minimal augmentation (close to original data)
    
    Design: Only apply necessary normalization, preserve original distribution characteristics
    """
    if dataset_type == 'cifar':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    elif dataset_type == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_nu_prior_augmentation(dataset_type='cifar', strategy='diverse'):
    """
    Prior augmentation for ν: Encode human prior knowledge about test distribution
    
    Design philosophy:
    - If we expect certain distribution shift in test set, simulate it here
    - Examples: viewpoint changes, lighting changes, partial occlusion, style transfer, etc.
    
    Args:
        dataset_type: Dataset type
        strategy: Augmentation strategy
            - 'diverse': Diverse augmentation (default)
            - 'robust': Robustness augmentation (stronger)
            - 'ood': Out-of-distribution augmentation (strongest)
    """
    if dataset_type == 'cifar':
        if strategy == 'diverse':
            # Strategy 1: Diverse augmentation (moderate)
            return transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),  # Larger crop range
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.6, 0.6, 0.6, 0.2)  # Stronger color jitter
                ], p=0.8),
                transforms.RandomGrayscale(p=0.3),  # Higher probability of grayscale
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        
        elif strategy == 'robust':
            # Strategy 2: Robustness augmentation (strong)
            return transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),  # Extreme crop
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),  # Add rotation
                transforms.RandomApply([
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.3)
                ], p=0.9),
                transforms.RandomGrayscale(p=0.4),
                transforms.RandomApply([GaussianBlur([0.1, 3.0])], p=0.7),
                transforms.ToTensor(),
                # Random noise addition (implemented in Dataset)
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        
        elif strategy == 'ood':
            # Strategy 3: Out-of-distribution augmentation (very strong)
            return transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),  # Large angle rotation
                transforms.RandomApply([
                    transforms.ColorJitter(1.0, 1.0, 1.0, 0.5)  # Very strong color jitter
                ], p=1.0),
                transforms.RandomGrayscale(p=0.5),  # 50% grayscale
                transforms.RandomApply([GaussianBlur([0.1, 4.0])], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        elif strategy == 'None':
            return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
                )
            ])
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    elif dataset_type == 'mnist':
        if strategy == 'diverse':
            return transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
                transforms.RandomRotation(15),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        elif strategy == 'robust':
            return transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.4, 1.0)),
                transforms.RandomRotation(25),
                transforms.RandomApply([GaussianBlur([0.1, 3.0])], p=0.7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        elif strategy == 'ood':
            return transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.3, 1.0)),
                transforms.RandomRotation(40),
                transforms.RandomApply([GaussianBlur([0.1, 4.0])], p=0.9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_val_transforms(dataset_type='cifar'):
    """
    Validation set augmentation: Only normalization (no data augmentation)
    """
    if dataset_type == 'cifar':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    elif dataset_type == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")