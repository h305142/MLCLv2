# losses/adadro_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import LossConfig


class AdaDROLoss(nn.Module):
    """
    AdaDRO Loss Function
    
    Core idea:
    1. P_tr (ptr) is the training distribution
    2. ν̃ (nu, after filtering) is the reference distribution
    3. Compute transport cost from P_tr to ν̃
    4. Construct worst-case distribution Q^λ on ν̃
    5. Minimize E_{p~P_tr}[E_{q~Q^λ}[ℓ(q)]]
    """
    
    def __init__(self, config: LossConfig):
        super(AdaDROLoss, self).__init__()
        # Use correct parameter names
        self.lambda_dro = config.lambda_dro
        self.epsilon = config.epsilon
        self.semantic_weight = config.semantic_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def compute_ce_loss(self, logits, labels, use_soft_labels=False):
        """
        Compute cross-entropy loss, supports both hard and soft labels
        
        Args:
            logits: [N, K] - predicted logits
            labels: [N] or [N, K] or dict - hard labels, soft labels, or Mixup dict
            use_soft_labels: bool - whether to use soft labels
        
        Returns:
            loss: [N] - loss for each sample
        """
        # Handle Mixup dictionary format
        if isinstance(labels, dict):
            # Mixup case: extract soft labels from dictionary
            lam = labels['lam']
            target1 = labels['target1']
            target2 = labels['target2']
            
            # Construct soft labels
            num_classes = logits.size(1)
            batch_size = logits.size(0)
            soft_labels = torch.zeros(batch_size, num_classes, device=logits.device)
            
            for i in range(batch_size):
                soft_labels[i, target1[i]] = lam[i]
                soft_labels[i, target2[i]] = 1 - lam[i]
            
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(soft_labels * log_probs).sum(dim=1)
        
        elif use_soft_labels and labels.dim() == 2:
            # Soft labels: use KL divergence
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(labels * log_probs).sum(dim=1)
        
        else:
            # Hard labels: use cross-entropy
            loss = self.ce_loss(logits, labels)
        
        return loss
        
    def compute_transport_costs(self, ptr_features, ptr_labels, 
                                nu_features, nu_labels, 
                                classifier_weights):
        """
        Compute transport cost matrix from P_tr to ν̃
        
        Args:
            ptr_features: [N, d] - features of P_tr
            ptr_labels: [N] - labels of P_tr
            nu_features: [M, d] - features of ν̃
            nu_labels: [M] or [M, K] or dict - labels of ν̃ (hard, soft, or Mixup)
            classifier_weights: [K, d] - classifier weights
        
        Returns:
            C_total: [N, M] - total transport cost
            C_X: [N, M] - feature space cost
            C_Y: [N, M] - label space cost
        """
        N = ptr_features.size(0)
        M = nu_features.size(0)
        
        # 1. Feature space cost
        feature_similarities = F.cosine_similarity(
            ptr_features.unsqueeze(1),
            nu_features.unsqueeze(0),
            dim=2
        )
        C_X = 1.0 - feature_similarities
        
        # 2. Label space cost
        # Handle ptr_labels (always hard labels)
        ptr_label_embeddings = classifier_weights[ptr_labels]  # [N, d]
        
        # Handle three cases for nu_labels
        if isinstance(nu_labels, dict):
            # Mixup dictionary
            lam = nu_labels['lam']
            target1 = nu_labels['target1']
            target2 = nu_labels['target2']
            
            embed1 = classifier_weights[target1]  # [M, d]
            embed2 = classifier_weights[target2]  # [M, d]
            nu_label_embeddings = lam.unsqueeze(1) * embed1 + (1 - lam.unsqueeze(1)) * embed2
        
        elif nu_labels.dim() == 2:
            # Soft labels
            nu_label_embeddings = torch.matmul(nu_labels, classifier_weights)  # [M, d]
        
        else:
            # Hard labels
            nu_label_embeddings = classifier_weights[nu_labels]  # [M, d]
        
        label_similarities = F.cosine_similarity(
            ptr_label_embeddings.unsqueeze(1),
            nu_label_embeddings.unsqueeze(0),
            dim=2
        )
        C_Y = 1.0 - label_similarities
        
        # 3. Total transport cost
        C_total = C_X + C_Y
        
        return C_total, C_X, C_Y
    
    def compute_worst_case_distribution(self, nu_losses, transport_costs):
        """
        Compute worst-case distribution Q^λ
        
        According to paper equation (6):
        Q^λ(q) ∝ E_{p~P_tr}[exp((ℓ(q) - λC(p,q)) / (λε)) / Z(p)]
        
        Args:
            nu_losses: [M] - loss for each sample in ν̃
            transport_costs: [N, M] - transport cost from P_tr to ν̃
        
        Returns:
            worst_case_probs: [M] - probability distribution on ν̃
        """
        N, M = transport_costs.shape
        
        # Step 1: Compute exponent term
        exponent = (nu_losses.unsqueeze(0) - self.lambda_dro * transport_costs) / (
            self.lambda_dro * self.epsilon + 1e-8
        )
        
        # Step 2: Numerical stability
        exponent_max = exponent.max(dim=1, keepdim=True)[0]
        exponent_stable = exponent - exponent_max
        exp_terms = torch.exp(exponent_stable)
        
        # Step 3: Compute normalization constant Z(p)
        Z_p = exp_terms.sum(dim=1, keepdim=True)
        
        # Step 4: Normalize
        normalized_weights = exp_terms / (Z_p + 1e-8)
        
        # Step 5: Expectation over P_tr
        worst_case_probs = normalized_weights.mean(dim=0)
        
        # Step 6: Re-normalize
        worst_case_probs = worst_case_probs / (worst_case_probs.sum() + 1e-8)
        
        return worst_case_probs
    
    def forward(self, 
                ptr_features, ptr_logits, ptr_labels,
                nu_features, nu_logits, nu_labels,
                classifier_weights, 
                moco_logits=None, moco_labels=None,
                use_soft_labels=False):
        """
        Forward pass
        
        Args:
            ptr_features: [N, d] - features of P_tr
            ptr_logits: [N, K] - classification logits of P_tr
            ptr_labels: [N] - labels of P_tr
            nu_features: [M, d] - features of ν̃ (after filtering)
            nu_logits: [M, K] - classification logits of ν̃
            nu_labels: [M] or [M, K] or dict - labels of ν̃ (hard, soft, or Mixup)
            classifier_weights: [K, d] - classifier weight matrix
            moco_logits: Optional, logits for MoCo contrastive learning
            moco_labels: Optional, labels for MoCo
            use_soft_labels: bool - whether to use soft labels (deprecated, auto-detected)
        
        Returns:
            loss_dict: Dictionary containing various losses and statistics
        """
        N = ptr_features.size(0)
        M = nu_features.size(0)
        
        # Automatically detect if soft labels
        is_soft_labels = isinstance(nu_labels, dict) or (
            isinstance(nu_labels, torch.Tensor) and nu_labels.dim() == 2
        )
        
        # 1. Compute loss for each sample in ν̃
        nu_losses = self.compute_ce_loss(nu_logits, nu_labels, use_soft_labels=is_soft_labels)  # [M]
        
        # 2. Compute transport cost from P_tr to ν̃
        C_total, C_X, C_Y = self.compute_transport_costs(
            ptr_features, ptr_labels,
            nu_features, nu_labels,
            classifier_weights
        )
        
        # 3. Compute worst-case distribution Q^λ
        worst_case_probs = self.compute_worst_case_distribution(
            nu_losses, C_total
        )
        
        # 4. Compute DRO loss
        dro_loss = (worst_case_probs * nu_losses).sum()
        
        # 5. Semantic contrastive loss
        semantic_loss = torch.tensor(0.0, device=dro_loss.device)
        if moco_logits is not None and moco_labels is not None:
            semantic_loss = F.cross_entropy(moco_logits, moco_labels)
        
        # 6. Total loss
        total_loss = dro_loss + self.semantic_weight * semantic_loss
        
        # 7. Return detailed information
        return {
            'total_loss': total_loss,
            'dro_loss': dro_loss,
            'semantic_loss': semantic_loss,
            'worst_case_probs': worst_case_probs.detach(),
            'transport_costs': C_total.mean().detach(),
            'C_X': C_X.mean().detach(),
            'C_Y': C_Y.mean().detach(),
            'ptr_size': N,
            'nu_size': M,
        }