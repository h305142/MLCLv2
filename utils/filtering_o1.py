# utils/filtering.py
import torch
import torch.nn.functional as F
from config.base_config import FilterConfig


class ReferenceDistributionFilter:
    """
    Reference distribution filter for unpaired samples
    
    Correct logic:
    - ptr_features and nu_features are independent (not paired)
    - For each nu[i], find its maximum matching confidence with all ptr[j]
    - If max_j sim(nu[i], ptr[j]) is too low, nu[i] is a "low-value sample"
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.global_weight = config.global_weight
        self.adaptive = config.adaptive
        
    def compute_matching_confidence(self, ptr_features, nu_features):
        """
        Compute maximum matching confidence for each sample in nu with ptr
        
        Args:
            ptr_features: [N_ptr, d] - features of P_tr
            nu_features: [N_nu, d] - features of nu (not paired with ptr)
        
        Returns:
            max_confidences: [N_nu] - maximum matching confidence for each nu sample
            similarities: [N_nu, N_ptr] - complete similarity matrix
        """
        ptr_norm = F.normalize(ptr_features, dim=1)  # [N_ptr, d]
        nu_norm = F.normalize(nu_features, dim=1)    # [N_nu, d]
        
        # similarities[i, j] = cosine similarity between nu[i] and ptr[j]
        similarities = torch.mm(nu_norm, ptr_norm.t())
        
        # max_confidences[i] = max_j sim(nu[i], ptr[j])
        max_confidences, _ = similarities.max(dim=1)  # [N_nu]
        
        return max_confidences, similarities
    
    def compute_adaptive_threshold(self, max_confidences, similarities=None):
        """
        Compute adaptive threshold
        
        Method 1 (recommended): tau = omega * mean(max_confidences) + (1-omega) * median(max_confidences)
        Method 2 (paper): consider local capacity
        """
        tau_global = max_confidences.mean()
        
        if self.adaptive and similarities is not None:
            # local_capacities[i] = sum_j sim(nu[i], ptr[j])
            local_capacities = similarities.sum(dim=1)  # [N_nu]
            
            max_capacity = local_capacities.max()
            normalized_capacities = local_capacities / (max_capacity + 1e-8)
            
            # Adaptive threshold: samples with larger capacity have lower thresholds
            adaptive_thresholds = tau_global * normalized_capacities
            
            return adaptive_thresholds
        else:
            # Method 1: global mean + median (more robust)
            tau_median = max_confidences.median()
            threshold = (
                self.global_weight * tau_global + 
                (1 - self.global_weight) * tau_median
            )
            return threshold * torch.ones_like(max_confidences)
    
    def filter_samples(self, ptr_features, nu_features, model=None, return_details=False):
        """
        Filter samples in nu
        
        Args:
            ptr_features: [N_ptr, d] - features of P_tr
            nu_features: [N_nu, d] - features of nu
        
        Returns:
            mask: [N_nu] - boolean mask
        """
        N_nu = nu_features.size(0)
        N_ptr = ptr_features.size(0)
        
        with torch.no_grad():
            if N_nu <= 1 or N_ptr == 0:
                mask = torch.ones(N_nu, dtype=torch.bool, device=nu_features.device)
                if return_details:
                    return mask, {
                        'max_confidences': torch.ones(N_nu, device=nu_features.device),
                        'adaptive_thresholds': torch.ones(N_nu, device=nu_features.device),
                        'filtered_ratio': 1.0,
                        'num_kept': N_nu,
                        'num_total': N_nu,
                        'warning': 'Boundary case'
                    }
                return mask
            
            max_confidences, similarities = self.compute_matching_confidence(
                ptr_features, nu_features
            )
            
            adaptive_thresholds = self.compute_adaptive_threshold(
                max_confidences, similarities
            )
            
            if self.adaptive:
                mask = max_confidences >= adaptive_thresholds
            else:
                mask = max_confidences >= self.confidence_threshold
            
            # Safety mechanism: keep at least 10% of samples
            if mask.float().mean() < 0.1:
                num_keep = max(1, int(N_nu * 0.1))
                top_indices = torch.argsort(max_confidences, descending=True)[:num_keep]
                mask = torch.zeros(N_nu, dtype=torch.bool, device=nu_features.device)
                mask[top_indices] = True
        
        if return_details:
            return mask, {
                'max_confidences': max_confidences.detach().cpu(),
                'adaptive_thresholds': adaptive_thresholds.detach().cpu() if isinstance(adaptive_thresholds, torch.Tensor) else adaptive_thresholds,
                'similarities': similarities.detach().cpu(),
                'filtered_ratio': mask.float().mean().item(),
                'num_kept': mask.sum().item(),
                'num_total': N_nu,
                'N_ptr': N_ptr
            }
        
        return mask
    
    def update_reference_distribution(self, original_distribution, filter_mask):
        """
        Update reference distribution (maintain normalization)
        """
        filtered_distribution = original_distribution * filter_mask.float()
        
        normalization = filtered_distribution.sum() + 1e-8
        filtered_distribution = filtered_distribution / normalization
        
        return filtered_distribution


class DynamicFilteringScheduler:
    """
    Dynamic filtering scheduler (cosine annealing)
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
        
        import math
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