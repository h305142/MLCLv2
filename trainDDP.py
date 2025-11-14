import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time
import argparse




try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        # Import from writer submodule for older PyTorch versions
        from torch.utils.tensorboard.writer import SummaryWriter
    except Exception:
        # Fallback to tensorboardX
        from tensorboardX import SummaryWriter




from models.resnet import ResNetEncoder
from models.ama_moco import AMA_MoCo
from data.dataset import get_dataloader
from config import Config








def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)








def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()








def main(rank, world_size):
    """
    Main training function
    Args:
        rank: Current process GPU ID (0, 1, 2, ...)
        world_size: Total number of GPUs
    """
    # DDP initialization
    setup_ddp(rank, world_size)
    is_main_process = (rank == 0)  # Only rank 0 prints logs and saves models
    
    cfg = Config()
    
    # Print configuration only in main process
    if is_main_process:
        print("=" * 60)
        print("AMA-MoCo Training Configuration (DDP)")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Dataset: {cfg.dataset}")
        print(f"Architecture: {cfg.arch}")
        print(f"Feature dimension: {cfg.dim}")
        print(f"Queue size: {cfg.K}")
        print(f"Multi-scale clustering: {cfg.scales}")
        print(f"Batch size per GPU: {cfg.batch_size}")
        print(f"Total batch size: {cfg.batch_size * world_size}")
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
    
    model = model.to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if is_main_process:
        print(f"\nModel created on {world_size} GPUs")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Data loading (DDP version)
    train_dataset = get_dataloader(
        cfg.dataset, 
        cfg.batch_size, 
        cfg.num_workers,
        return_dataset=True  # Requires modification in dataset.py to support returning dataset
    )
    
    # Use DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr * world_size,  # Linear scaling of learning rate
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs * len(train_loader)
    )
    
    # TensorBoard (main process only)
    if is_main_process:
        log_dir = f'runs/ama_moco_{cfg.dataset}_{cfg.arch}_ddp'
        writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard logs: {log_dir}")
    
    # Training loop
    if is_main_process:
        print("\n" + "=" * 60)
        print("Start training...")
        print("=" * 60 + "\n")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        # Set different random seed for each epoch
        train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for i, (images, _) in enumerate(train_loader):
            im_q, im_k = images
            im_q = im_q.to(rank, non_blocking=True)
            im_k = im_k.to(rank, non_blocking=True)
            
            # Forward pass
            loss = model(im_q, im_k)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update queues (access DDP internal module)
            model.module.update_queues()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Print logs only in main process
            if is_main_process and i % cfg.print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{cfg.epochs}] "
                      f"Step [{i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {current_lr:.6f}")
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('LR', current_lr, global_step)
        
        # Synchronize loss across all processes
        avg_loss = epoch_loss / len(train_loader)
        avg_loss_tensor = torch.tensor(avg_loss, device=rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / world_size).item()
        
        # Epoch statistics (main process only)
        if is_main_process:
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
                    'model_state_dict': model.module.state_dict(),  # Save internal model
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
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': cfg.__dict__
                }, f'checkpoints/ama_moco_{cfg.dataset}_epoch_{epoch+1}.pth')
                print(f"Checkpoint saved at epoch {epoch+1}\n")
    
    if is_main_process:
        print("=" * 60)
        print("Training finished!")
        print(f"Best loss: {best_loss:.4f}")
        print("=" * 60)
        writer.close()
    
    # Clean up DDP
    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Launch multi-process
    torch.multiprocessing.spawn(
        main,
        args=(args.world_size,),
        nprocs=args.world_size,
        join=True
    )