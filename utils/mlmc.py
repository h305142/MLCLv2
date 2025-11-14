# utils/mlmc.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from config.base_config import MLMCConfig


class MLMCGradientEstimator:
    """
    Multi-Level Monte Carlo (MLMC) Gradient Estimator
    
    Core idea:
    E[Y] ≈ E[Y_0] + Σ_{l=1}^L E[Y_l - Y_{l-1}]
    
    where Y_l is the estimate using 2^l samples
    """
    
    def __init__(self, config: MLMCConfig):
        self.config = config
        self.max_level = config.max_level
        self.a = config.a  # bias decay: bias ≈ 2^(-a*l)
        self.b = config.b  # variance decay: var ≈ 2^(-b*l)
        self.c = config.c  # cost growth: cost ≈ 2^(c*l)
        
        # Statistics
        self.level_stats = {
            'samples': [],     # Number of samples per level
            'variances': [],   # Variance per level
            'costs': []        # Computational cost per level
        }
        
    def compute_level_approximation(
        self, 
        features: torch.Tensor,      # [N, d]
        labels: torch.Tensor,         # [N]
        classifier_weights: torch.Tensor,  # [K, d]
        level: int,
        return_loss: bool = False
    ) -> torch.Tensor:
        """
        Compute level l approximation
        
        Use N_l = 2^level samples
        """
        # 1. Sampling
        num_samples = min(2**level, features.size(0))
        
        if num_samples < features.size(0):
            indices = torch.randperm(features.size(0), device=features.device)[:num_samples]
            sampled_features = features[indices]
            sampled_labels = labels[indices]
        else:
            sampled_features = features
            sampled_labels = labels
        
        # 2. Compute cost matrix
        # Feature space cost
        feature_similarities = F.cosine_similarity(
            sampled_features.unsqueeze(1),  # [N_l, 1, d]
            sampled_features.unsqueeze(0),  # [1, N_l, d]
            dim=2
        )  # [N_l, N_l]
        C_X = 1.0 - feature_similarities
        
        # Label space cost
        label_embeddings = classifier_weights[sampled_labels]  # [N_l, d]
        label_similarities = F.cosine_similarity(
            label_embeddings.unsqueeze(1),  # [N_l, 1, d]
            label_embeddings.unsqueeze(0),  # [1, N_l, d]
            dim=2
        )  # [N_l, N_l]
        C_Y = 1.0 - label_similarities
        
        # Total cost
        C_total = C_X + C_Y  # [N_l, N_l]
        
        # 3. Compute loss
        logits = torch.mm(sampled_features, classifier_weights.t())  # [N_l, K]
        losses = F.cross_entropy(logits, sampled_labels, reduction='none')  # [N_l]
        
        # 4. Compute worst-case weights
        # w_ij = exp((ℓ_i - λ*C_ij) / (λ*ε))
        weights = torch.exp(
            (losses.unsqueeze(1) - self.config.lambda_reg * C_total) / 
            (self.config.lambda_reg * self.config.epsilon)
        )  # [N_l, N_l]
        
        # Normalize
        Z = weights.sum(dim=1, keepdim=True) + 1e-8  # [N_l, 1]
        normalized_weights = weights / Z  # [N_l, N_l]
        
        # Worst-case weight for each sample (average over j)
        worst_case_weights = normalized_weights.mean(dim=1)  # [N_l]
        
        # 5. Worst-case objective
        objective = (worst_case_weights * losses).sum()
        
        if return_loss:
            return objective, losses
        return objective
    
    def compute_optimal_samples(
        self, 
        target_epsilon: float,
        variance_estimates: Optional[list] = None
    ) -> list:
        """
        Compute optimal number of samples per level
        
        According to MLMC theory:
        N_l ≈ ε^(-2) * sqrt(V_l / C_l) * Σ_l sqrt(V_l * C_l)
        
        where:
        - V_l: variance at level l
        - C_l: computational cost at level l
        """
        L = self.max_level
        
        if variance_estimates is None:
            # Use theoretical estimates
            variance_estimates = [2**(-self.b * l) for l in range(L + 1)]
        
        # Compute costs
        costs = [2**(self.c * l) for l in range(L + 1)]
        
        # Compute sum term
        sum_sqrt_VC = sum(np.sqrt(v * c) for v, c in zip(variance_estimates, costs))
        
        # Compute samples per level
        optimal_samples = []
        for l in range(L + 1):
            V_l = variance_estimates[l]
            C_l = costs[l]
            
            N_l = (target_epsilon ** (-2)) * np.sqrt(V_l / C_l) * sum_sqrt_VC
            optimal_samples.append(max(1, int(np.ceil(N_l))))
        
        return optimal_samples
    
    def estimate_gradient(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        model: torch.nn.Module,
        target_epsilon: float = 1e-3,
        use_adaptive_sampling: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        MLMC gradient estimation
        
        Args:
            features: [N, d] - features
            labels: [N] - labels
            model: model (must have get_classifier_weights method)
            target_epsilon: target accuracy
            use_adaptive_sampling: whether to use adaptive sampling
        
        Returns:
            gradients: dict - gradient estimates for each parameter
            info: dict - statistics
        """
        # 1. Determine number of levels
        L = min(int(np.ceil(np.log2(1.0 / target_epsilon))), self.max_level)
        
        # 2. Get classifier weights
        classifier_weights = model.get_classifier_weights()
        
        # 3. Compute sampling probabilities (importance sampling)
        # Use bias and variance estimates to adjust probabilities
        if use_adaptive_sampling and len(self.level_stats['variances']) > 0:
            # Based on historical variance
            probs = []
            for l in range(L + 1):
                if l < len(self.level_stats['variances']):
                    # Higher variance, higher sampling probability
                    prob = np.sqrt(self.level_stats['variances'][l])
                else:
                    # Theoretical estimate
                    prob = 2**(-self.b * l / 2)
                probs.append(prob)
            probs = np.array(probs)
            probs = probs / probs.sum()
        else:
            # Uniform sampling
            probs = np.ones(L + 1) / (L + 1)
        
        # 4. Randomly select level
        level = np.random.choice(L + 1, p=probs)
        
        # 5. Zero gradients
        model.zero_grad()
        
        # 6. Compute gradient estimate
        if level == 0:
            # Base case: Y_0
            objective_0 = self.compute_level_approximation(
                features, labels, classifier_weights, level=0
            )
            
            # Backward pass
            objective_0.backward()
            
            # Collect gradients (multiply by 1/p_0 for importance sampling correction)
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone() / probs[0]
            
            variance = 0.0
            
        else:
            # Difference estimator: Y_l - Y_{l-1}
            # Compute Y_l
            model.zero_grad()
            objective_l = self.compute_level_approximation(
                features, labels, classifier_weights, level=level
            )
            objective_l.backward()
            
            grads_l = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads_l[name] = param.grad.clone()
            
            # Compute Y_{l-1}
            model.zero_grad()
            objective_l_minus_1 = self.compute_level_approximation(
                features, labels, classifier_weights, level=level - 1
            )
            objective_l_minus_1.backward()
            
            grads_l_minus_1 = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads_l_minus_1[name] = param.grad.clone()
            
            # Compute difference gradient (multiply by 1/p_l for importance sampling correction)
            gradients = {}
            for name in grads_l.keys():
                gradients[name] = (grads_l[name] - grads_l_minus_1[name]) / probs[level]
            
            # Estimate variance (for adaptation)
            diff = objective_l.item() - objective_l_minus_1.item()
            variance = diff ** 2
        
        # 7. Update statistics
        if level >= len(self.level_stats['variances']):
            self.level_stats['variances'].extend([0.0] * (level + 1 - len(self.level_stats['variances'])))
            self.level_stats['samples'].extend([0] * (level + 1 - len(self.level_stats['samples'])))
        
        # Update variance using moving average
        alpha = 0.1  # Smoothing coefficient
        old_var = self.level_stats['variances'][level]
        self.level_stats['variances'][level] = (1 - alpha) * old_var + alpha * variance
        self.level_stats['samples'][level] += 1
        
        # 8. Return info
        info = {
            'level': level,
            'probability': probs[level],
            'variance': variance,
            'num_levels': L + 1,
            'level_probabilities': probs.tolist()
        }
        
        return gradients, info


class AdaptiveMLMC(MLMCGradientEstimator):
    """
    Adaptive MLMC Gradient Estimator
    
    Dynamically adjusts based on runtime statistics:
    1. Maximum level L
    2. Sampling probability per level
    3. Target accuracy
    """
    
    def __init__(self, config: MLMCConfig):
        super().__init__(config)
        self.target_epsilon = config.epsilon
        
        # Historical statistics
        self.bias_history = []
        self.variance_history = []
        self.mse_history = []
        
        # Adaptive parameters
        self.adaptation_interval = 10  # Adjust every N iterations
        self.iteration = 0
        
    def update_statistics(self, gradients: Dict[str, torch.Tensor], info: Dict):
        """
        Update statistics
        
        Args:
            gradients: gradient estimates
            info: estimation info
        """
        # Estimate bias (using gradient norm as proxy)
        grad_norm = sum(g.norm().item() ** 2 for g in gradients.values()) ** 0.5
        estimated_bias = grad_norm * (2 ** (-self.a * info['level']))
        
        # Estimate variance
        estimated_variance = info['variance']
        
        # MSE = bias^2 + variance
        mse = estimated_bias ** 2 + estimated_variance
        
        # Record history
        self.bias_history.append(estimated_bias)
        self.variance_history.append(estimated_variance)
        self.mse_history.append(mse)
        
        # Limit history length
        max_history = 100
        if len(self.bias_history) > max_history:
            self.bias_history = self.bias_history[-max_history:]
            self.variance_history = self.variance_history[-max_history:]
            self.mse_history = self.mse_history[-max_history:]
    
    def adapt_parameters(self):
        """
        Adaptively adjust parameters
        """
        if len(self.bias_history) < 10:
            return
        
        # Use recent statistics
        recent_bias = np.mean(self.bias_history[-5:])
        recent_variance = np.mean(self.variance_history[-5:])
        recent_mse = np.mean(self.mse_history[-5:])
        
        # 1. Adjust maximum level
        # If bias is too large, increase levels
        if recent_bias > self.target_epsilon / 2:
            self.max_level = min(10, self.max_level + 1)
            print(f"[MLMC] Increasing max_level to {self.max_level} (bias={recent_bias:.6f})")
        
        # If bias is very small and variance is also small, can reduce levels
        elif recent_bias < self.target_epsilon / 10 and recent_variance < self.target_epsilon / 10:
            self.max_level = max(1, self.max_level - 1)
            print(f"[MLMC] Decreasing max_level to {self.max_level} (bias={recent_bias:.6f})")
        
        # 2. Adjust target accuracy
        # If MSE is much smaller than target, can relax accuracy for efficiency
        if recent_mse < (self.target_epsilon ** 2) / 4:
            self.target_epsilon = min(1e-2, self.target_epsilon * 1.2)
            print(f"[MLMC] Relaxing epsilon to {self.target_epsilon:.6f}")
        
        # If MSE is too large, need to tighten accuracy
        elif recent_mse > (self.target_epsilon ** 2) * 2:
            self.target_epsilon = max(1e-4, self.target_epsilon * 0.8)
            print(f"[MLMC] Tightening epsilon to {self.target_epsilon:.6f}")
    
    def estimate_gradient_adaptive(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        model: torch.nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Adaptive gradient estimation
        
        Args:
            features: [N, d] - features
            labels: [N] - labels
            model: model
        
        Returns:
            gradients: dict - gradient estimates
            info: dict - statistics
        """
        # 1. Estimate gradient
        gradients, info = self.estimate_gradient(
            features, labels, model, 
            target_epsilon=self.target_epsilon,
            use_adaptive_sampling=True
        )
        
        # 2. Update statistics
        self.update_statistics(gradients, info)
        
        # 3. Adaptive adjustment
        self.iteration += 1
        if self.iteration % self.adaptation_interval == 0:
            self.adapt_parameters()
        
        # 4. Add adaptive info
        info.update({
            'target_epsilon': self.target_epsilon,
            'max_level': self.max_level,
            'recent_bias': np.mean(self.bias_history[-5:]) if len(self.bias_history) >= 5 else 0,
            'recent_variance': np.mean(self.variance_history[-5:]) if len(self.variance_history) >= 5 else 0
        })
        
        return gradients, info


# Utility functions
def apply_mlmc_gradients(
    model: torch.nn.Module,
    gradients: Dict[str, torch.Tensor],
    learning_rate: float = 1e-3
):
    """
    Apply MLMC gradient update
    
    Args:
        model: model
        gradients: MLMC estimated gradients
        learning_rate: learning rate
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in gradients:
                param.data -= learning_rate * gradients[name]


def visualize_mlmc_statistics(estimator: MLMCGradientEstimator, save_path=None):
    """
    Visualize MLMC statistics
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Subplot 1: Variance per level
    if estimator.level_stats['variances']:
        axes[0].bar(range(len(estimator.level_stats['variances'])), 
                   estimator.level_stats['variances'])
        axes[0].set_xlabel('Level')
        axes[0].set_ylabel('Variance')
        axes[0].set_title('Variance per Level')
        axes[0].set_yscale('log')
    
    # Subplot 2: Samples per level
    if estimator.level_stats['samples']:
        axes[1].bar(range(len(estimator.level_stats['samples'])), 
                   estimator.level_stats['samples'])
        axes[1].set_xlabel('Level')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Samples per Level')
    
    # Subplot 3: Bias and variance history (AdaptiveMLMC only)
    if isinstance(estimator, AdaptiveMLMC) and estimator.bias_history:
        axes[2].plot(estimator.bias_history, label='Bias', alpha=0.7)
        axes[2].plot(estimator.variance_history, label='Variance', alpha=0.7)
        axes[2].axhline(estimator.target_epsilon, color='r', linestyle='--', label='Target ε')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Value')
        axes[2].set_title('Bias and Variance History')
        axes[2].set_yscale('log')
        axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()