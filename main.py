# main.py
import torch
import argparse
import os
import sys
from pathlib import Path
import datetime

from config.base_config import get_default_config
from config.dataset_configs import (
    get_cifar10_config, get_cifar100_config, 
    get_colored_mnist_config, get_waterbirds_config
)
from models.adadro_model import AdaDROModel
from data.loaders import create_data_loaders
from training.trainer import AdaDROTrainer
from utils.misc import set_seed, setup_directories


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='AdaDRO: Adaptive Distributionally Robust Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'colored_mnist', 'waterbirds'],
                       help='Dataset to use')
    
    # Model
    parser.add_argument('--arch', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained model')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--semantic-epochs', type=int, default=50,
                       help='Number of semantic calibration epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    
    # DRO Parameters
    parser.add_argument('--lambda-reg', type=float, default=1.0,
                       help='DRO regularization parameter')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Entropic regularization')
    parser.add_argument('--kappa', type=float, default=1.0,
                       help='Semantic loss weight')
    
    # Filtering
    parser.add_argument('--filter-threshold', type=float, default=0,
                       help='Confidence threshold')
    parser.add_argument('--similarity-threshold', type=float, default=0.15,
                       help='Similarity threshold')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Data loading workers')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                       help='Pin memory')
    
    # Paths
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume checkpoint path')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluation only mode')
    
    # Advanced
    parser.add_argument('--amp', action='store_true',
                       help='Use mixed precision')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Checkpoint save frequency')
    parser.add_argument('--print-freq', type=int, default=50,
                       help='Print frequency')
    
    return parser.parse_args()


def get_config_from_args(args):
    """Build configuration from arguments"""
    
    # Select base config by dataset
    config_map = {
        'cifar10': get_cifar10_config,
        'cifar100': get_cifar100_config,
        'colored_mnist': get_colored_mnist_config,
        'waterbirds': get_waterbirds_config
    }
    
    config = config_map.get(args.dataset, get_default_config)()
    
    # Update model config
    config.model.arch = args.arch
    config.model.pretrained = args.pretrained
    
    # Update training config
    config.training.epochs = args.epochs
    config.training.semantic_epochs = args.semantic_epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.num_workers = args.num_workers
    config.training.pin_memory = args.pin_memory
    
    # Update loss config
    config.loss.lambda_reg = args.lambda_reg
    config.loss.epsilon = args.epsilon
    config.loss.kappa = args.kappa
    
    # Update filter config
    config.filter.confidence_threshold = args.filter_threshold
    config.filter.similarity_threshold = args.similarity_threshold
    
    # Update system config
    config.seed = args.seed
    config.save_dir = args.save_dir
    config.log_dir = args.log_dir
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config.experiment_name = f'{args.dataset}_{args.arch}_lambda{args.lambda_reg}_{timestamp}'
    else:
        config.experiment_name = args.experiment_name
    
    # Set device
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device
    
    # Advanced settings
    config.amp = args.amp
    config.grad_clip = args.grad_clip
    config.save_freq = args.save_freq
    config.print_freq = args.print_freq
    
    return config, args


def print_configuration(config, args):
    """Print configuration details"""
    print("\n" + "="*80)
    print("AdaDRO: Adaptive Distributionally Robust Optimization")
    print("="*80)
    
    print("\n[Dataset Configuration]")
    print(f"  Dataset:           {config.dataset}")
    print(f"  Number of Classes: {config.model.num_classes}")
    print(f"  Data Directory:    {args.data_dir}")
    
    print("\n[Model Configuration]")
    print(f"  Architecture:      {config.model.arch}")
    print(f"  Feature Dim:       {config.model.feature_dim}")
    print(f"  Pretrained:        {config.model.pretrained}")
    
    print("\n[Training Configuration]")
    print(f"  Total Epochs:      {config.training.epochs}")
    print(f"  Semantic Epochs:   {config.training.semantic_epochs}")
    print(f"  Batch Size:        {config.training.batch_size}")
    print(f"  Learning Rate:     {config.training.learning_rate}")
    print(f"  Weight Decay:      {config.training.weight_decay}")
    print(f"  Mixed Precision:   {config.amp}")
    
    print("\n[DRO Parameters]")
    print(f"  Lambda:            {config.loss.lambda_reg}")
    print(f"  Epsilon:           {config.loss.epsilon}")
    print(f"  Kappa:             {config.loss.kappa}")
    
    print("\n[Filtering Configuration]")
    print(f"  Confidence Thresh: {config.filter.confidence_threshold}")
    print(f"  Similarity Thresh: {config.filter.similarity_threshold}")
    
    print("\n[System Configuration]")
    print(f"  Device:            {config.device}")
    print(f"  Random Seed:       {config.seed}")
    print(f"  Num Workers:       {config.training.num_workers}")
    
    print("\n[Save Configuration]")
    print(f"  Experiment Name:   {config.experiment_name}")
    print(f"  Save Directory:    {config.save_dir}")
    print(f"  Log Directory:     {config.log_dir}")
    
    if args.resume:
        print(f"\n[Resume]")
        print(f"  Checkpoint:        {args.resume}")
    
    print("\n" + "="*80 + "\n")


def validate_config(config, args):
    """Validate configuration"""
    errors = []
    
    # Check device availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        errors.append("CUDA not available but device set to 'cuda'")
    
    # Check parameter ranges
    if config.loss.lambda_reg <= 0:
        errors.append(f"lambda_reg must be positive, got {config.loss.lambda_reg}")
    
    if config.loss.epsilon <= 0:
        errors.append(f"epsilon must be positive, got {config.loss.epsilon}")
    
    if not (0 <= config.filter.confidence_threshold <= 1):
        errors.append(f"confidence_threshold must be in [0,1], got {config.filter.confidence_threshold}")
    
    # Check paths
    if args.resume and not os.path.exists(args.resume):
        errors.append(f"Checkpoint not found: {args.resume}")
    
    if args.evaluate and not args.resume:
        errors.append("--evaluate requires --resume")
    
    if errors:
        print("\n[Configuration Errors]")
        for error in errors:
            print(f"  - {error}")
        print()
        sys.exit(1)


def load_checkpoint(model, trainer, checkpoint_path):
    """Load checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Model state loaded")
        
        if 'optimizer_state_dict' in checkpoint and trainer.optimizer:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Optimizer state loaded")
        
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  Scheduler state loaded")
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"  Resuming from epoch {epoch}")
        print(f"  Best validation accuracy: {metrics.get('val_accuracy', 0):.2f}%")
        
        return epoch, metrics
    
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        sys.exit(1)


def main():
    """Main function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Build configuration
    config, args = get_config_from_args(args)
    
    # Print configuration
    print_configuration(config, args)
    
    # Validate configuration
    validate_config(config, args)
    
    # Set random seed
    print("Setting random seed...")
    set_seed(config.seed)
    
    # Setup directories
    print("Setting up directories...")
    setup_directories(config.save_dir, config.log_dir)
    
    # Load data
    print("\nLoading data...")
    try:
        train_loader, val_loader = create_data_loaders(config)
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create model
    print("\nCreating model...")
    try:
        model = AdaDROModel(config.model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Setup trainer
    print("\nSetting up trainer...")
    try:
        trainer = AdaDROTrainer(config, train_loader, val_loader, model)
    except Exception as e:
        print(f"Error setting up trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, trainer, args.resume)
    
    # Evaluation mode
    if args.evaluate:
        print("\n[Evaluation Mode]")
        val_metrics = trainer.validate()
        print("\n" + "="*60)
        print("Evaluation Results:")
        print(f"  Validation Loss:     {val_metrics['val_loss']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['val_accuracy']:.2f}%")
        print("="*60)
        return
    
    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    try:
        best_accuracy = trainer.train()
        
        print("\n" + "="*80)
        print("Training completed successfully!")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")
        print(f"Checkpoints saved in: {config.save_dir}")
        print(f"Logs saved in: {config.log_dir}")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=start_epoch,
            metrics={'interrupted': True},
            is_best=False
        )
        print("Checkpoint saved")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()