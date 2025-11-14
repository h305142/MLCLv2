import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        # Import from writer submodule for older PyTorch versions
        from torch.utils.tensorboard.writer import SummaryWriter
    except Exception:
        # Fallback to tensorboardX
        from tensorboardX import SummaryWriter
import os
import time

from models.resnet import ResNetEncoder
from models.ama_moco import AMA_MoCo
from data.dataset import get_dataloader
from config import Config


def main():
    cfg = Config()
    
    # Print configuration
    print("=" * 60)
    print("AMA-MoCo Training Configuration")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset}")
    print(f"Architecture: {cfg.arch}")
    print(f"Feature dimension: {cfg.dim}")
    print(f"Queue size: {cfg.K}")
    print(f"Multi-scale clustering: {cfg.scales}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Epochs: {cfg.epochs}")
    print("=" * 60)
    
    # Create model
    encoder = ResNetEncoder(base_model=cfg.arch, dim=cfg.dim)
    model = AMA_MoCo(
        encoder=encoder,
        dim=cfg.dim,
        K=cfg.K,
        m=cfg.m,
        T=cfg.T,
        T_local=cfg.T_local,
        scales=cfg.scales,
        weight_momentum=cfg.weight_momentum,
        cluster_update_freq=cfg.cluster_update_freq
    )
    
    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nModel created on {device}")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Data loading using unified interface
    train_loader = get_dataloader(cfg.dataset, cfg.batch_size, cfg.num_workers)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs * len(train_loader)
    )
    
    # TensorBoard
    log_dir = f'runs/ama_moco_{cfg.dataset}_{cfg.arch}'
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard logs: {log_dir}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Start training...")
    print("=" * 60 + "\n")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for i, (images, _) in enumerate(train_loader):
            im_q, im_k = images  # Two augmented versions
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)
            
            # Forward pass
            loss = model(im_q, im_k)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update queues after backward pass
            model.update_queues()  # Separate method
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Print logs
            if i % cfg.print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{cfg.epochs}] "
                      f"Step [{i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {current_lr:.6f}")
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('LR', current_lr, global_step)
                
                # # Visualize cluster weights
                # for scale in cfg.scales:
                #     scale_name = f'scale_{scale}'
                #     weights = model._get_cluster_weights(scale_name).detach().cpu().numpy()
                #     writer.add_histogram(f'Weights/{scale_name}', weights, global_step)
        
        # Epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{cfg.epochs}] Summary")
        print(f"{'='*60}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"{'='*60}\n")
        
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': cfg.__dict__
            }, f'checkpoints/ama_moco_{cfg.dataset}_best.pth')
            print(f"Best model saved! (Loss: {best_loss:.4f})\n")
        
        # Save checkpoint periodically
        if (epoch + 1) % cfg.save_freq == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': cfg.__dict__
            }, f'checkpoints/ama_moco_{cfg.dataset}_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}\n")
    
    print("=" * 60)
    print("Training finished!")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)
    
    writer.close()


if __name__ == '__main__':
    main()