import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any
import torch
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.csv")
        self.config_file = os.path.join(self.experiment_dir, "config.json")
        self.log_file = os.path.join(self.experiment_dir, "training.log")
        
        self.metrics_history = []
        
        self._setup_csv_headers()
    
    def _setup_csv_headers(self):
        """Set up CSV headers for metrics file"""
        headers = [
            'epoch', 'total_loss', 'dro_loss', 'semantic_loss', 'accuracy',
            'val_loss', 'val_accuracy', 'filtered_ratio', 'epoch_time'
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_config(self, config: Any):
        """Log configuration to JSON file"""
        if hasattr(config, '__dict__'):
            config_dict = self._config_to_dict(config)
        else:
            config_dict = config
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert config object to dictionary recursively"""
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._config_to_dict(value)
                else:
                    result[key] = value
            return result
        return config
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for current epoch"""
        self.metrics_history.append({'epoch': epoch, **metrics})
        
        row = [
            epoch,
            metrics.get('total_loss', 0),
            metrics.get('dro_loss', 0),
            metrics.get('semantic_loss', 0),
            metrics.get('accuracy', 0),
            metrics.get('val_loss', 0),
            metrics.get('val_accuracy', 0),
            metrics.get('filtered_ratio', 0),
            metrics.get('epoch_time', 0)
        ]
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        log_message = f"Epoch {epoch}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log_message(log_message)
    
    def log_message(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def plot_training_curves(self):
        """Plot training curves for all metrics"""
        if len(self.metrics_history) == 0:
            return
        
        epochs = [m['epoch'] for m in self.metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {self.experiment_name}')
        
        # Loss curves
        train_losses = [m.get('total_loss', 0) for m in self.metrics_history]
        val_losses = [m.get('val_loss', 0) for m in self.metrics_history]
        axes[0, 0].plot(epochs, train_losses, label='Train Loss')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        train_accs = [m.get('accuracy', 0) for m in self.metrics_history]
        val_accs = [m.get('val_accuracy', 0) for m in self.metrics_history]
        axes[0, 1].plot(epochs, train_accs, label='Train Accuracy')
        axes[0, 1].plot(epochs, val_accs, label='Val Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Component losses
        dro_losses = [m.get('dro_loss', 0) for m in self.metrics_history]
        semantic_losses = [m.get('semantic_loss', 0) for m in self.metrics_history]
        axes[1, 0].plot(epochs, dro_losses, label='DRO Loss')
        axes[1, 0].plot(epochs, semantic_losses, label='Semantic Loss')
        axes[1, 0].set_title('Component Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Filtering ratio
        filtered_ratios = [m.get('filtered_ratio', 0) for m in self.metrics_history]
        axes[1, 1].plot(epochs, filtered_ratios)
        axes[1, 1].set_title('Filtering Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'training_curves.png'))
        plt.close()
    
    def save_final_results(self, best_metrics: Dict[str, float]):
        """Save final results and plot training curves"""
        results = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'best_metrics': best_metrics,
            'total_epochs': len(self.metrics_history)
        }
        
        with open(os.path.join(self.experiment_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.plot_training_curves()


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.available = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.available = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.available:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalar values"""
        if self.available:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of tensor values"""
        if self.available:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close TensorBoard writer"""
        if self.available:
            self.writer.close()