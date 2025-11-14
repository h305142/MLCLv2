from .base_config import ExperimentConfig, ModelConfig, TrainingConfig, get_default_config


def get_colored_mnist_config() -> ExperimentConfig:
    config = get_default_config()
    config.dataset = 'colored_mnist'
    config.model.arch = 'simple'  
    config.model.num_classes = 10
    config.model.feature_dim = 128
    config.training.batch_size = 512
    config.training.epochs = 100
    config.training.learning_rate = 0.001  
    return config

def get_cifar10_config() -> ExperimentConfig:
    config = get_default_config()
    config.dataset = 'cifar10'
    config.model.num_classes = 10
    config.training.batch_size = 256
    return config

def get_cifar100_config() -> ExperimentConfig:
    config = get_default_config()
    config.dataset = 'cifar100'
    config.model.num_classes = 100
    config.training.batch_size = 128
    config.training.learning_rate = 0.005
    return config

def get_waterbirds_config() -> ExperimentConfig:
    config = get_default_config()
    config.dataset = 'waterbirds'
    config.model.num_classes = 2
    config.training.batch_size = 64
    config.training.learning_rate = 0.001
    return config