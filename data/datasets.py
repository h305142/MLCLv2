# data/datasets.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import datasets
import numpy as np
import random
from PIL import Image


# ========================================
# 1. Colored MNIST Base Dataset (keep unchanged)
# ========================================

class ColoredMNIST(Dataset):
    """Colored MNIST Dataset (same as before)"""
    
    def __init__(self, root, train=True, download=True, transform=None,
                 color_correlation=0.9):
        self.mnist = datasets.MNIST(root, train=train, download=download, transform=None)
        self.transform = transform
        self.color_correlation = color_correlation
        
        print(f"✓ ColoredMNIST initialization:")
        print(f"  Dataset: {'Training set' if train else 'Test set'}")
        print(f"  Color correlation: {color_correlation:.1%}")
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, index):
        img, label = self.mnist[index]
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        is_odd = label % 2 == 1
        
        if random.random() < self.color_correlation:
            use_red = is_odd
        else:
            use_red = random.random() < 0.5
        
        h, w = img_array.shape
        colored_img = np.zeros((h, w, 3), dtype=np.float32)
        
        if use_red:
            colored_img[:, :, 0] = img_array
            colored_img[:, :, 1] = img_array * 0.2
            colored_img[:, :, 2] = img_array * 0.2
        else:
            colored_img[:, :, 0] = img_array * 0.2
            colored_img[:, :, 1] = img_array
            colored_img[:, :, 2] = img_array * 0.2
        
        colored_img = (colored_img * 255).astype(np.uint8)
        colored_img = Image.fromarray(colored_img, mode='RGB')
        
        if self.transform is not None:
            colored_img = self.transform(colored_img)
        
        return colored_img, label


# ========================================
# 2. Four-View Dataset (new design)
# ========================================

class FourViewDataset(Dataset):
    """
    Four-View Dataset (AdaDRO specific)
    
    Design:
    1. im_q, im_k: Contrastive learning views (standard MoCo augmentation)
    2. ptr: P_tr samples (minimal augmentation)
    3. nu: ν samples (prior augmentation, encoding human expectation of test distribution)
    
    Parameters:
    - mixup_nu: Whether to mix Mixup samples in ν
    - noise_nu: Whether to add noise in ν
    """
    
    def __init__(self, base_dataset, 
                 contrastive_transform_q,  # Augmentation for im_q
                 contrastive_transform_k,  # Augmentation for im_k
                 ptr_transform,            # Augmentation for ptr
                 nu_transform,             # Prior augmentation for nu
                 mixup_nu=False,           # Whether to use Mixup in ν
                 mixup_alpha=0.2,
                 mixup_ratio=0.3,
                 noise_nu=False,           # Whether to add noise in ν
                 noise_std=0.1,
                 noise_ratio=0.2):
        """
        Args:
            base_dataset: Base dataset
            contrastive_transform_q: Contrastive learning augmentation for im_q
            contrastive_transform_k: Contrastive learning augmentation for im_k
            ptr_transform: Minimal augmentation for ptr
            nu_transform: Prior augmentation for nu
            mixup_nu: Whether to mix Mixup samples in ν
            mixup_alpha: Beta distribution parameter for Mixup
            mixup_ratio: Ratio of Mixup samples
            noise_nu: Whether to add noise in ν
            noise_std: Noise standard deviation
            noise_ratio: Ratio of noise samples
        """
        self.base_dataset = base_dataset
        self.contrastive_transform_q = contrastive_transform_q
        self.contrastive_transform_k = contrastive_transform_k
        self.ptr_transform = ptr_transform
        self.nu_transform = nu_transform
        
        self.mixup_nu = mixup_nu
        self.mixup_alpha = mixup_alpha
        self.mixup_ratio = mixup_ratio
        
        self.noise_nu = noise_nu
        self.noise_std = noise_std
        self.noise_ratio = noise_ratio
        
        print(f"✓ FourViewDataset initialization:")
        print(f"  Contrastive learning: im_q, im_k (standard MoCo augmentation)")
        print(f"  DRO distributions: ptr (minimal augmentation), nu (prior augmentation)")
        if mixup_nu:
            print(f"  Mixup ratio in ν: {mixup_ratio:.1%}")
        if noise_nu:
            print(f"  Noise ratio in ν: {noise_ratio:.1%}")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        # 1. Get original data
        img, target = self.base_dataset[index]
        
        # 2. Contrastive learning views (im_q, im_k)
        im_q = self.contrastive_transform_q(img)
        im_k = self.contrastive_transform_k(img)
        
        # 3. DRO distribution: ptr (minimal augmentation)
        ptr = self.ptr_transform(img)
        
        # 4. DRO distribution: nu (prior augmentation + optional Mixup/noise)
        if self.mixup_nu and random.random() < self.mixup_ratio:
            # Apply Mixup
            nu, target_nu, is_mixup = self._apply_mixup(img, target)
        else:
            # Standard prior augmentation
            nu = self.nu_transform(img)
            target_nu = target
            is_mixup = False
        
        # Optional: add noise
        if self.noise_nu and random.random() < self.noise_ratio:
            noise = torch.randn_like(nu) * self.noise_std
            nu = torch.clamp(nu + noise, 0, 1)
        
        # 5. Return four views
        return (im_q, im_k, ptr, nu), target, target_nu, is_mixup
    
    def _apply_mixup(self, img, target):
        """Apply Mixup data augmentation"""
        # Randomly select another sample
        mix_index = random.randint(0, len(self.base_dataset) - 1)
        img_mix, target_mix = self.base_dataset[mix_index]
        
        # Generate Mixup coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Pixel-space mixing
        img_tensor = TF.to_tensor(img)
        img_mix_tensor = TF.to_tensor(img_mix)
        nu_tensor = lam * img_tensor + (1 - lam) * img_mix_tensor
        
        # Convert back to PIL and apply prior augmentation
        nu = TF.to_pil_image(nu_tensor)
        nu = self.nu_transform(nu)
        
        # Return soft label tuple
        target_nu = (lam, target, target_mix)
        
        return nu, target_nu, True


# ========================================
# 3. Colored MNIST Four-View Wrapper
# ========================================

class ColoredMNISTFourView(Dataset):
    """
    Four-view version of Colored MNIST
    
    Special prior augmentation: color flipping (simulating anti-correlation in test set)
    """
    
    def __init__(self, base_colored_mnist,
                 contrastive_transform_q,
                 contrastive_transform_k,
                 ptr_transform,
                 nu_transform,
                 color_flip_ratio=0.5):
        """
        Args:
            base_colored_mnist: ColoredMNIST dataset instance
            contrastive_transform_q: Contrastive learning augmentation for im_q
            contrastive_transform_k: Contrastive learning augmentation for im_k
            ptr_transform: Minimal augmentation for ptr
            nu_transform: Prior augmentation for nu
            color_flip_ratio: Ratio of color-flipped samples in ν
        """
        self.base_dataset = base_colored_mnist
        self.contrastive_transform_q = contrastive_transform_q
        self.contrastive_transform_k = contrastive_transform_k
        self.ptr_transform = ptr_transform
        self.nu_transform = nu_transform
        self.color_flip_ratio = color_flip_ratio
        
        print(f"✓ ColoredMNISTFourView initialization:")
        print(f"  Color flip ratio: {color_flip_ratio:.1%}")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        
        # Contrastive learning views
        im_q = self.contrastive_transform_q(img)
        im_k = self.contrastive_transform_k(img)
        
        # ptr: minimal augmentation
        ptr = self.ptr_transform(img)
        
        # nu: prior augmentation + optional color flipping
        if random.random() < self.color_flip_ratio:
            # Color flip (encoding anti-correlation prior in test set)
            img_flipped = self._flip_color(img)
            nu = self.nu_transform(img_flipped)
            is_flipped = True
        else:
            nu = self.nu_transform(img)
            is_flipped = False
        
        return (im_q, im_k, ptr, nu), target, target, is_flipped
    
    def _flip_color(self, img):
        """Flip color channels (red ↔ green)"""
        img_tensor = TF.to_tensor(img)
        img_flipped = img_tensor.clone()
        # Swap red and green channels
        img_flipped[0], img_flipped[1] = img_tensor[1], img_tensor[0]
        return TF.to_pil_image(img_flipped)