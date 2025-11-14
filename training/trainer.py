# training/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from typing import Dict, Any


from models.adadro_model import AdaDROModel
from losses.adadro_loss import AdaDROLoss
from utils.filtering_o1 import ReferenceDistributionFilter, DynamicFilteringScheduler
from utils.mlmc import MLMCGradientEstimator
from utils.metrics import MetricsCalculator
from utils.logging import Logger
from config.base_config import ExperimentConfig


class AdaDROTrainer:
    """
    AdaDRO Trainer
    
    Core design:
    1. im_q, im_k -> contrastive learning views (MoCo)
    2. ptr -> P_tr (training distribution, minimal augmentation)
    3. nu -> ν (prior distribution, strong augmentation)
    4. ν̃: filtered ν
    5. Compute transport cost from P_tr to ν̃
    6. Construct worst-case distribution Q^λ
    """
    
    def __init__(self, config: ExperimentConfig, train_loader: DataLoader, 
                 val_loader: DataLoader, model: AdaDROModel):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(config.device)
        
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_utilities()
        self.setup_logging()
        
    def setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.training.epochs
        )
        
    def setup_criterion(self):
        self.criterion = AdaDROLoss(self.config.loss).to(self.config.device)
        
    def setup_utilities(self):
        self.filter = ReferenceDistributionFilter(self.config.filter)
        self.filter_scheduler = DynamicFilteringScheduler(
            total_epochs=self.config.training.epochs
        )
        self.mlmc = MLMCGradientEstimator(self.config.mlmc)
        self.metrics = MetricsCalculator()
        
    def setup_logging(self):
        self.logger = Logger(
            log_dir=self.config.log_dir,
            experiment_name=self.config.experiment_name
        )
        
    def semantic_calibration_phase(self):
        """
        Phase 1: Semantic calibration phase
        Objective: Unsupervised feature learning for representation learning
        
        Fix: Use four-view data format
        """
        print("=" * 60)
        print("Starting semantic calibration phase...")
        print("=" * 60)
        
        self.model.train()
        
        for epoch in range(self.config.training.semantic_epochs):
            total_loss = 0.0
            total_moco_loss = 0.0
            total_ce_loss = 0.0
            
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Fix: Correctly unpack four-view data
                # batch_data = (views, target_hard, target_nu, flag)
                views, target_hard, target_nu, flag = batch_data
                im_q, im_k, ptr, nu = views
                
                # Move to device
                im_q = im_q.to(self.config.device)
                im_k = im_k.to(self.config.device)
                target = target_hard.to(self.config.device)
                
                # 1. Contrastive learning loss (MoCo)
                moco_logits, moco_labels = self.model.moco_forward(im_q, im_k)
                moco_loss = F.cross_entropy(moco_logits, moco_labels)
                
                # 2. Classification loss (using im_q)
                features, logits = self.model(im_q)
                ce_loss = F.cross_entropy(logits, target)
                
                # 3. Total loss
                total_loss_batch = moco_loss + ce_loss
                
                # 4. Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                # 5. Accumulate losses
                total_loss += total_loss_batch.item()
                total_moco_loss += moco_loss.item()
                total_ce_loss += ce_loss.item()
                
                # 6. Print progress
                if batch_idx % 50 == 0:
                    print(f'[Semantic] Epoch {epoch+1}/{self.config.training.semantic_epochs}, '
                          f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'MoCo: {moco_loss.item():.4f}, CE: {ce_loss.item():.4f}')
            
            # Epoch summary
            avg_loss = total_loss / len(self.train_loader)
            avg_moco = total_moco_loss / len(self.train_loader) 
            avg_ce = total_ce_loss / len(self.train_loader)
            
            print(f'[Semantic] Epoch {epoch+1} Summary:')
            print(f'  Total Loss: {avg_loss:.4f}')
            print(f'  MoCo Loss: {avg_moco:.4f}')
            print(f'  CE Loss: {avg_ce:.4f}')
            print('-' * 60)
        
        print("Semantic calibration phase completed!")
        print("=" * 60)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'dro_loss': 0.0,
            'semantic_loss': 0.0,
            'filtered_ratio': 0.0,
            'flag_ratio': 0.0,
            'accuracy': 0.0,
            'transport_cost': 0.0
        }
        
        num_batches = 0
        correct = 0
        total = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # batch_data = (views, target_hard, target_nu, flag)
            metrics = self.train_step(batch_data, batch_data[1], epoch)
            
            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            
            if 'correct' in metrics:
                correct += metrics['correct']
                total += metrics['total']
            
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'[Train] Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}:')
                print(f'  Total Loss: {metrics.get("total_loss", 0):.4f}')
                print(f'  DRO Loss: {metrics.get("dro_loss", 0):.4f}')
                print(f'  Filtered Ratio: {metrics.get("filtered_ratio", 0):.2%}')
                print(f'  Flag Ratio: {metrics.get("flag_ratio", 0):.2%}')
        
        # Calculate average metrics
        for key in epoch_metrics:
            if key != 'accuracy':
                epoch_metrics[key] /= num_batches
        
        if total > 0:
            epoch_metrics['accuracy'] = 100.0 * correct / total
        
        self.scheduler.step()
        
        return epoch_metrics
    
    def train_step(self, batch_data, target, epoch):
        """Train one batch"""
        # Unpack four-view data
        views, target_hard, target_nu, flag = batch_data
        im_q, im_k, ptr, nu = views
        
        device = self.config.device
        
        # Move to device
        im_q = im_q.to(device)
        im_k = im_k.to(device)
        ptr = ptr.to(device)
        nu = nu.to(device)
        label_ptr = target_hard.to(device)
        flag = flag.to(device)
        
        batch_size = label_ptr.size(0)
        
        # 1. Handle ν labels
        if isinstance(target_nu, dict):
            label_nu = {
                'lam': target_nu['lam'].to(device),
                'target1': target_nu['target1'].to(device),
                'target2': target_nu['target2'].to(device)
            }
            use_soft_labels = True
        else:
            label_nu = target_nu.to(device)
            use_soft_labels = False
        
        # 2. Contrastive learning
        moco_logits, moco_labels = self.model.moco_forward(im_q, im_k)
        moco_loss = F.cross_entropy(moco_logits, moco_labels)
        
        # 3. Extract features
        ptr_features, ptr_logits = self.model(ptr)
        nu_features, nu_logits = self.model(nu)
        
        # 4. Filtering
        filter_mask, filter_details = self.filter.filter_samples(
            ptr_features,
            nu_features,
            return_details=True
        )
        
        if filter_mask.sum() == 0:
            print(f"Warning: All samples filtered out!")
            self.optimizer.zero_grad()
            moco_loss.backward()
            self.optimizer.step()
            
            return {
                'total_loss': moco_loss.item(),
                'dro_loss': 0.0,
                'semantic_loss': moco_loss.item(),
                'transport_cost': 0.0,
                'filtered_ratio': 1.0,
                'flag_ratio': flag.float().mean().item(),
                'correct': 0,
                'total': 0
            }
        
        # 5. Construct ν̃
        nu_tilde_features = nu_features[filter_mask]
        nu_tilde_logits = nu_logits[filter_mask]
        
        if isinstance(label_nu, dict):
            nu_tilde_labels = {
                'lam': label_nu['lam'][filter_mask],
                'target1': label_nu['target1'][filter_mask],
                'target2': label_nu['target2'][filter_mask]
            }
        else:
            nu_tilde_labels = label_nu[filter_mask]
        
        # 6. AdaDRO loss
        classifier_weights = self.model.get_classifier_weights()
        
        loss_dict = self.criterion(
            ptr_features=ptr_features,
            ptr_logits=ptr_logits,
            ptr_labels=label_ptr,
            nu_features=nu_tilde_features,
            nu_logits=nu_tilde_logits,
            nu_labels=nu_tilde_labels,
            classifier_weights=classifier_weights,
            moco_logits=moco_logits,
            moco_labels=moco_labels,
            use_soft_labels=use_soft_labels
        )
        
        total_loss = loss_dict['total_loss']
        
        # 7. Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 8. Statistics
        pred = ptr_logits.argmax(dim=1)
        correct = (pred == label_ptr).sum().item()
        
        return {
            'total_loss': total_loss.item(),
            'dro_loss': loss_dict['dro_loss'].item(),
            'semantic_loss': loss_dict['semantic_loss'].item() if isinstance(
                loss_dict['semantic_loss'], torch.Tensor
            ) else loss_dict['semantic_loss'],
            'transport_cost': loss_dict.get('transport_costs', 0.0),
            'filtered_ratio': filter_details['filtered_ratio'],
            'flag_ratio': flag.float().mean().item(),
            'correct': correct,
            'total': batch_size
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validation function
        
        Note: Validation set uses standard data format (data, target), not four-view format
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # Handle validation set simple format
                # PyTorch DataLoader returns list by default, convert to tuple
                if isinstance(batch_data, list):
                    batch_data = tuple(batch_data)
                
                if len(batch_data) != 2:
                    raise ValueError(
                        f"Validation batch should have 2 elements (data, target), "
                        f"got {len(batch_data)}"
                    )
                
                data, target = batch_data
                
                # Move to device
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                # Forward pass
                _, logits = self.model(data)
                loss = F.cross_entropy(logits, target)
                
                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_accuracy': 100.0 * correct / total
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved! Val Accuracy: {metrics['val_accuracy']:.2f}%")
    
    def train(self) -> float:
        """Complete training process"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Phase 1: Semantic calibration
        self.semantic_calibration_phase()
        
        # Phase 2: AdaDRO training
        print("\n" + "=" * 60)
        print("Starting adaptive DRO training phase...")
        print("=" * 60)
        
        best_val_acc = 0.0
        
        for epoch in range(self.config.training.epochs):
            start_time = time.time()
            
            updated_config = self.filter_scheduler.update_filter_config(
                self.config.filter, epoch
            )
            self.filter = ReferenceDistributionFilter(updated_config)
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            epoch_time = time.time() - start_time
            all_metrics = {**train_metrics, **val_metrics, 'epoch_time': epoch_time}
            
            is_best = val_metrics['val_accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['val_accuracy']
            
            self.save_checkpoint(epoch, all_metrics, is_best)
            self.logger.log_metrics(epoch, all_metrics)
            
            print("\n" + "=" * 60)
            print(f'Epoch {epoch+1}/{self.config.training.epochs} Summary:')
            print("=" * 60)
            print(f'Training:')
            print(f'  Total Loss:     {train_metrics["total_loss"]:.4f}')
            print(f'  DRO Loss:       {train_metrics["dro_loss"]:.4f}')
            print(f'  Semantic Loss:  {train_metrics["semantic_loss"]:.4f}')
            print(f'  Accuracy:       {train_metrics["accuracy"]:.2f}%')
            print(f'  Filtered Ratio: {train_metrics["filtered_ratio"]:.2%}')
            print(f'  Flag Ratio:     {train_metrics["flag_ratio"]:.2%}')
            print(f'\nValidation:')
            print(f'  Loss:           {val_metrics["val_loss"]:.4f}')
            print(f'  Accuracy:       {val_metrics["val_accuracy"]:.2f}%')
            print(f'\nBest Val Acc:     {best_val_acc:.2f}%')
            print(f'Epoch Time:       {epoch_time:.2f}s')
            print(f'Learning Rate:    {self.optimizer.param_groups[0]["lr"]:.6f}')
            print("=" * 60 + "\n")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print("=" * 60)
        
        return best_val_acc