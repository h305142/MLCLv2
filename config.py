class Config:
    # Model parameters
    arch = 'resnet18'  # Options: 'resnet18' or 'resnet50'
    dim = 128
    K = 4096  # Queue size
    m = 0.999  # Momentum coefficient
    T = 0.07  # Global contrastive temperature
    T_local = 0.2  # Local contrastive temperature
    scales = [10, 30, 100]  # Multi-scale clustering numbers
    alpha = 0.1  # Weight update learning rate
    weight_momentum = 0.9  # Weight update momentum
    cluster_update_freq = 100  # Cluster update frequency
    
    # Training parameters
    dataset = 'cifar100'  # Options: 'cifar10', 'cifar100', or 'stl10'
    batch_size = 256
    epochs = 200
    lr = 0.03
    weight_decay = 1e-4
    momentum = 0.9
    
    # Other settings
    num_workers = 4
    print_freq = 10
    save_freq = 10
    
    # Dataset-specific parameters
    dataset_configs = {
        'cifar10': {
            'num_classes': 10,
            'image_size': 32,
            'channels': 3
        },
        'cifar100': {
            'num_classes': 100,
            'image_size': 32,
            'channels': 3
        },
        'stl10': {
            'num_classes': 10,
            'image_size': 96,
            'channels': 3
        }
    }
    
    def get_dataset_info(self):
        """Get configuration info for current dataset"""
        return self.dataset_configs.get(self.dataset, {})