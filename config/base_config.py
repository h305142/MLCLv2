# config/base_config.py
import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    arch: str = 'resnet18'
    num_classes: int = 10
    feature_dim: int = 512
    moco_dim: int = 128
    moco_k: int = 4096
    moco_m: float = 0.999
    moco_t: float = 0.07
    use_mlp: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 200
    semantic_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.01
    weight_decay: float = 5e-6
    momentum: float = 0.9
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LossConfig:
    """Loss function configuration"""
    # AdaDRO core parameters
    lambda_dro: float = 1.0           # DRO loss weight
    epsilon: float = 0.1              # Wasserstein radius
    semantic_weight: float = 0.5      # MoCo semantic loss weight
    
    # Prior augmentation strategy for ν (core new feature)
    nu_augmentation_strategy: str = 'diverse'  # 'diverse', 'robust', 'ood'
    # - 'diverse': Diverse augmentation (moderate, suitable for general distribution shift)
    # - 'robust': Robustness augmentation (strong, suitable for larger distribution shift)
    # - 'ood': Out-of-distribution augmentation (very strong, suitable for severe distribution shift)
    
    # Mixup configuration in ν
    mixup_nu: bool = False            # Whether to mix Mixup samples in ν
    mixup_alpha: float = 0.2          # Beta distribution parameter for Mixup
    mixup_ratio: float = 0.3          # Ratio of Mixup samples in ν
    
    # Noise configuration in ν
    noise_nu: bool = False            # Whether to add noise in ν
    noise_std: float = 0.1            # Gaussian noise standard deviation
    noise_ratio: float = 0.2          # Ratio of samples with added noise
    
    # Colored MNIST specific configuration
    color_flip_ratio: float = 0.5     # Ratio of color-flipped samples in ν
    
    # Transport cost weights
    feature_cost_scale: float = 1.0   # Feature space cost weight
    label_cost_scale: float = 0.5     # Label space cost weight
    
    # Deprecated configuration (kept for backward compatibility)
    reference_strategy: str = 'combined'  # Deprecated, kept for compatibility
    noise_std_old: float = 0.1            # Old noise configuration (integrated into noise_nu)
    noise_ratio_old: float = 0.3


@dataclass
class FilterConfig:
    """Filter configuration"""
    enable_filtering: bool = False
    confidence_threshold: float = 0  # Confidence threshold
    global_weight: float = 0.6         # Global confidence weight
    adaptive: bool = True              # Whether to adaptively adjust threshold


@dataclass
class MLMCConfig:
    """Multi-level Monte Carlo configuration"""
    max_level: int = 5
    a: int = 2
    b: int = 1
    c: int = 1
    target_epsilon: float = 1e-3


@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    filter: FilterConfig
    mlmc: MLMCConfig
    dataset: str = 'cifar10'  # 'cifar10', 'cifar100', 'colored_mnist'
    device: str = 'cuda'
    seed: int = 42
    save_dir: str = './checkpoints'
    log_dir: str = './logs'
    experiment_name: str = 'adadro_experiment'
    
    def __post_init__(self):
        """Configuration validation and automatic adjustment"""
        # Automatically set number of classes based on dataset
        if self.dataset == 'cifar10':
            self.model.num_classes = 10
        elif self.dataset == 'cifar100':
            self.model.num_classes = 100
        elif self.dataset == 'colored_mnist':
            self.model.num_classes = 10
        
        # Validate ν augmentation strategy
        valid_strategies = ['diverse', 'robust', 'ood']
        if self.loss.nu_augmentation_strategy not in valid_strategies:
            raise ValueError(
                f"nu_augmentation_strategy must be one of {valid_strategies}, "
                f"got '{self.loss.nu_augmentation_strategy}'"
            )
        
        # Ensure device is available
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = 'cpu'


def get_default_config() -> ExperimentConfig:
    """Get default configuration"""
    return ExperimentConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        loss=LossConfig(),
        filter=FilterConfig(),
        mlmc=MLMCConfig()
    )


# ========================================
# Predefined Configuration Templates
# ========================================

def get_cifar10_config() -> ExperimentConfig:
    """CIFAR-10 recommended configuration"""
    config = get_default_config()
    config.dataset = 'cifar10'
    config.model.num_classes = 10
    config.loss.nu_augmentation_strategy = 'diverse'
    config.loss.mixup_nu = True
    config.loss.mixup_ratio = 0.3
    return config


def get_cifar100_config() -> ExperimentConfig:
    """CIFAR-100 recommended configuration"""
    config = get_default_config()
    config.dataset = 'cifar100'
    config.model.num_classes = 100
    config.loss.nu_augmentation_strategy = 'robust'  # Stronger augmentation
    config.loss.mixup_nu = True
    config.loss.mixup_ratio = 0.4
    return config


def get_colored_mnist_config() -> ExperimentConfig:
    """Colored MNIST recommended configuration"""
    config = get_default_config()
    config.dataset = 'colored_mnist'
    config.model.num_classes = 10
    config.loss.nu_augmentation_strategy = 'diverse'
    config.loss.color_flip_ratio = 0.5  # 50% color flip
    config.training.batch_size = 128    # Smaller batch size
    return config


def get_robust_config() -> ExperimentConfig:
    """Robustness experiment configuration (strong augmentation)"""
    config = get_default_config()
    config.loss.nu_augmentation_strategy = 'robust'
    config.loss.mixup_nu = True
    config.loss.mixup_ratio = 0.4
    config.loss.noise_nu = True
    config.loss.noise_ratio = 0.2
    config.loss.lambda_dro = 1.5  # Increase DRO weight
    return config


def get_ood_config() -> ExperimentConfig:
    """Out-of-distribution generalization experiment configuration (very strong augmentation)"""
    config = get_default_config()
    config.loss.nu_augmentation_strategy = 'ood'
    config.loss.mixup_nu = True
    config.loss.mixup_ratio = 0.5
    config.loss.noise_nu = True
    config.loss.noise_ratio = 0.3
    config.loss.lambda_dro = 2.0  # Further increase DRO weight
    config.loss.epsilon = 0.15    # Increase Wasserstein radius
    return config


# ========================================
# Configuration Loading and Saving Tools
# ========================================

def save_config(config: ExperimentConfig, path: str):
    """Save configuration to file"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {path}")


def load_config(path: str) -> ExperimentConfig:
    """Load configuration from file"""
    import json
    
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct nested dataclass
    config = ExperimentConfig(
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        loss=LossConfig(**config_dict['loss']),
        filter=FilterConfig(**config_dict['filter']),
        mlmc=MLMCConfig(**config_dict['mlmc']),
        dataset=config_dict['dataset'],
        device=config_dict['device'],
        seed=config_dict['seed'],
        save_dir=config_dict['save_dir'],
        log_dir=config_dict['log_dir'],
        experiment_name=config_dict['experiment_name']
    )
    
    print(f"Config loaded from {path}")
    return config


def print_config(config: ExperimentConfig):
    """Print configuration (beautified output)"""
    print("\n" + "=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    
    print(f"\nDataset: {config.dataset}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print(f"Experiment: {config.experiment_name}")
    
    print(f"\nModel:")
    print(f"  - Architecture: {config.model.arch}")
    print(f"  - Num Classes: {config.model.num_classes}")
    print(f"  - Feature Dim: {config.model.feature_dim}")
    print(f"  - MoCo Dim: {config.model.moco_dim}")
    print(f"  - MoCo K: {config.model.moco_k}")
    
    print(f"\nTraining:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Semantic Epochs: {config.training.semantic_epochs}")
    print(f"  - Batch Size: {config.training.batch_size}")
    print(f"  - Learning Rate: {config.training.learning_rate}")
    print(f"  - Weight Decay: {config.training.weight_decay}")
    
    print(f"\nLoss & DRO:")
    print(f"  - Lambda DRO: {config.loss.lambda_dro}")
    print(f"  - Epsilon (Wasserstein): {config.loss.epsilon}")
    print(f"  - Semantic Weight: {config.loss.semantic_weight}")
    
    print(f"\nPrior Augmentation for ν:")
    print(f"  - Strategy: {config.loss.nu_augmentation_strategy}")
    print(f"  - Mixup in ν: {config.loss.mixup_nu}")
    if config.loss.mixup_nu:
        print(f"    - Mixup Ratio: {config.loss.mixup_ratio}")
        print(f"    - Mixup Alpha: {config.loss.mixup_alpha}")
    print(f"  - Noise in ν: {config.loss.noise_nu}")
    if config.loss.noise_nu:
        print(f"    - Noise Ratio: {config.loss.noise_ratio}")
        print(f"    - Noise Std: {config.loss.noise_std}")
    if config.dataset == 'colored_mnist':
        print(f"  - Color Flip Ratio: {config.loss.color_flip_ratio}")
    
    print(f"\nFilter:")
    print(f"  - Confidence Threshold: {config.filter.confidence_threshold}")
    print(f"  - Global Weight: {config.filter.global_weight}")
    print(f"  - Adaptive: {config.filter.adaptive}")
    
    print("=" * 60 + "\n")