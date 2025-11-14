import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_directories(*dirs):
    """Create directories if they don't exist"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """Get available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, path):
    """Save model state dictionary to file"""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Load model state dictionary from file"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model