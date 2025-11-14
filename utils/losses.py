import torch
import torch.nn.functional as F


def contrastive_loss_vectorized(q, k, queue, temperature=0.07):
    """
    Global contrastive loss (standard InfoNCE)
    
    Args:
        q: [B, D] query features
        k: [B, D] positive key features  
        queue: [D, K] negative samples queue
        temperature: float
    
    Returns:
        loss: scalar
    """
    batch_size = q.size(0)
    
    # Positive sample logits: [B, 1]
    l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(1)
    
    # Negative sample logits: [B, K]
    l_neg = torch.einsum('bd,dk->bk', q, queue.detach())
    
    # Concatenate [B, 1+K]
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    
    # Labels are all 0 (positive sample at position 0)
    labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)
    
    return F.cross_entropy(logits, labels)


def masked_contrastive_loss_vectorized(
    q, k, queue, k_clusters, queue_clusters, 
    temperature=0.07, 
    return_per_sample=False
):
    """
    Local contrastive loss (samples within same cluster as negatives)
    
    Args:
        q: [B, D] query
        k: [B, D] positive key
        queue: [D, K] local queue
        k_clusters: [B] cluster labels for current batch
        queue_clusters: [K] cluster labels for each sample in queue
        temperature: float
        return_per_sample: bool, whether to return per-sample losses
    
    Returns:
        loss: scalar or [B]
    """
    batch_size = q.size(0)
    
    # Positive sample logits
    l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(1)  # [B, 1]
    
    # Negative sample logits
    l_neg = torch.einsum('bd,dk->bk', q, queue.detach())  # [B, K]
    
    # Build mask: keep only negatives from same cluster
    # masks[i, j] = 1 means sample i and queue[j] are in same cluster
    masks = (k_clusters.unsqueeze(1) == queue_clusters.unsqueeze(0)).float()  # [B, K]
    
    # Mask out samples from different clusters using very small values (weight ≈ 0 after softmax)
    l_neg_masked = torch.where(
        masks.bool(),
        l_neg,
        torch.full_like(l_neg, -1e10)
    )
    
    # Check for empty clusters (no samples from that cluster in queue)
    valid_mask = masks.sum(dim=1) > 0  # [B]
    
    if not valid_mask.all():
        # If some samples have no negatives from their cluster in queue, fallback to using entire queue
        l_neg_masked[~valid_mask] = l_neg[~valid_mask]
    
    # Concatenate logits
    logits = torch.cat([l_pos, l_neg_masked], dim=1) / temperature  # [B, 1+K]
    labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)
    
    if return_per_sample:
        return F.cross_entropy(logits, labels, reduction='none')  # [B]
    else:
        return F.cross_entropy(logits, labels)  # scalar