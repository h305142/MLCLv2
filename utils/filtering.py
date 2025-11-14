# utils/filtering.py
import torch
import torch.nn.functional as F
import math
from config.base_config import FilterConfig


class ReferenceDistributionFilter:
    """
    Reference distribution filter
    
    Based on the paper:
    1. Compute matching confidence from q (im_q) to k (im_k)
    2. Filter low-quality samples using adaptive threshold
    3. tau(q) = omega * tau_global + (1-omega) * tau_local(q)
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.global_weight = config.global_weight
        
    def compute_matching_confidence(self, features, augmented_features):
        """
        Compute matching confidence
        
        Args:
            features: [N, d] - features of im_q
            augmented_features: [N, d] - features of im_k
        
        Returns:
            confidences: [N] - matching confidence for each sample
            similarities: [N, N] - similarity matrix
        """
        # Normalize
        q = F.normalize(features, dim=1)
        k = F.normalize(augmented_features, dim=1)
        
        # Compute similarity matrix
        similarities = torch.mm(q, k.t())  # [N, N]
        
        # Diagonal elements = matching confidence for corresponding samples
        confidences = torch.diag(similarities)  # [N]
        
        return confidences, similarities
    
    def compute_adaptive_threshold(self, confidences, similarities):
        """
        Compute adaptive threshold
        
        tau(q) = omega * tau_global + (1-omega) * tau_local(q)
        """
        # 1. Global threshold
        tau_global = confidences.mean()
        
        # 2. Local threshold (excluding diagonal)
        N = similarities.size(0)
        
        if N <= 1:
            # Edge case: only one sample
            tau_local = confidences
        else:
            # Set diagonal to -inf, then take maximum
            similarities_masked = similarities.clone()
            similarities_masked.fill_diagonal_(-float('inf'))
            tau_local, _ = similarities_masked.max(dim=1)
        
        # 3. Adaptive threshold
        adaptive_thresholds = (
            self.global_weight * tau_global + 
            (1 - self.global_weight) * tau_local
        )
        
        return adaptive_thresholds
    
    def filter_samples(self, features, augmented_features, return_details=False):
        """
        Filter samples
        """
        N = features.size(0)
        
        # Filtering operation does not require gradient
        with torch.no_grad():
            # Edge condition: too few samples
            if N <= 1:
                mask = torch.ones(N, dtype=torch.bool, device=features.device)
                if return_details:
                    return mask, {
                        'confidences': torch.ones(N, device=features.device),
                        'adaptive_thresholds': torch.zeros(N, device=features.device),
                        'filtered_ratio': 1.0,
                        'num_kept': N,
                        'num_total': N,
                        'warning': 'N<=1, keeping all samples'
                    }
                return mask
            
            # Compute matching confidence
            confidences, similarities = self.compute_matching_confidence(
                features, augmented_features
            )
            
            # Compute adaptive threshold
            adaptive_thresholds = self.compute_adaptive_threshold(
                confidences, similarities
            )
            
            # Filter
            mask = confidences >= adaptive_thresholds
            
            # Apply minimum threshold
            if self.confidence_threshold > 0:
                min_threshold_mask = confidences >= self.confidence_threshold
                mask = mask & min_threshold_mask
            
            # Safety mechanism: keep at least 10% of samples
            min_keep_ratio = 0.1
            if mask.float().mean() < min_keep_ratio:
                num_keep = max(1, int(N * min_keep_ratio))
                top_indices = torch.argsort(confidences, descending=True)[:num_keep]
                mask = torch.zeros(N, dtype=torch.bool, device=features.device)
                mask[top_indices] = True
        
        if return_details:
            # Compute mean of local thresholds excluding diagonal
            similarities_masked = similarities.clone()
            similarities_masked.fill_diagonal_(-float('inf'))
            tau_local_mean = similarities_masked.max(dim=1)[0].mean().item()
            
            return mask, {
                'confidences': confidences.cpu(),
                'adaptive_thresholds': adaptive_thresholds.cpu(),
                'tau_global': confidences.mean().item(),
                'tau_local_mean': tau_local_mean,
                'filtered_ratio': mask.float().mean().item(),
                'num_kept': mask.sum().item(),
                'num_total': N
            }
        
        return mask


class DynamicFilteringScheduler:
    """
    Dynamic filtering scheduler
    """
    
    def __init__(self, initial_threshold=0.5, final_threshold=0.7, 
                 total_epochs=200, warmup_epochs=10):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
    def get_threshold(self, epoch):
        """
        Get threshold for current epoch (cosine annealing)
        """
        if epoch < self.warmup_epochs:
            return self.initial_threshold
        
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        # Use math library
        cos_progress = (1 - math.cos(progress * math.pi)) / 2
        threshold = (
            self.initial_threshold + 
            (self.final_threshold - self.initial_threshold) * cos_progress
        )
        
        return threshold
    
    def update_filter_config(self, filter_config, epoch):
        """
        Update filter configuration
        """
        new_threshold = self.get_threshold(epoch)
        filter_config.confidence_threshold = new_threshold
        return filter_config
    
    def get_schedule_info(self, total_epochs=None):
        """
        Get schedule information (for visualization)
        """
        if total_epochs is None:
            total_epochs = self.total_epochs
            
        epochs = list(range(total_epochs))
        thresholds = [self.get_threshold(e) for e in epochs]
        
        return {
            'epochs': epochs,
            'thresholds': thresholds,
            'initial': self.initial_threshold,
            'final': self.final_threshold,
            'warmup': self.warmup_epochs
        }