# models/moco.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MoCo(nn.Module):
    def __init__(self, encoder, feature_dim, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        Args:
            encoder: Shared backbone encoder
            feature_dim: Feature dimension output by encoder
            dim: Output dimension of projector
            K: Queue size
            m: Momentum coefficient
            T: Temperature
            mlp: Whether to use MLP projector (otherwise use linear layer)
        """
        super(MoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # Directly use the passed encoder (shared) - encoder_q is shared externally, do not reassign
        self.encoder_q = encoder
        
        # Create momentum encoder (deep copy)
        self.encoder_k = copy.deepcopy(encoder)
        
        # Stop gradients for encoder_k
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        if mlp:
            # 2-layer MLP projector
            self.projector_q = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, dim)
            )
            self.projector_k = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, dim)
            )
        else:
            # Linear projector
            self.projector_q = nn.Linear(feature_dim, dim)
            self.projector_k = nn.Linear(feature_dim, dim)
        
        # Initialize projector_k (copy from projector_q)
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        for param in self.projector_k.parameters():
            param.requires_grad = False
        
        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update for encoder_k and projector_k"""
        # Update encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        # Update projector_k
        for param_q, param_k in zip(self.projector_q.parameters(), 
                                     self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Ensure queue size is divisible by batch_size (optional, avoid overflow)
        assert self.K % batch_size == 0, f"Queue size {self.K} should be divisible by batch size {batch_size}"
        
        # Replace keys in queue
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """
        Args:
            im_q: Query images [B, C, H, W]
            im_k: Key images [B, C, H, W]
        
        Returns:
            logits: [B, K+1]
            labels: [B] (all zeros, positive sample at position 0)
        """
        # Query branch (with gradients)
        q_features = self.encoder_q.get_features(im_q)  # [B, feature_dim]
        q = self.projector_q(q_features)                # [B, dim]
        q = F.normalize(q, dim=1)
        
        # Key branch (no gradients)
        with torch.no_grad():
            # Momentum update
            self._momentum_update_key_encoder()
            
            k_features = self.encoder_k.get_features(im_k)  # [B, feature_dim]
            k = self.projector_k(k_features)                # [B, dim]
            k = F.normalize(k, dim=1)
        
        # Compute logits
        # Positive logits: [B, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: [B, K]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Concatenate: [B, K+1]
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        
        # Labels: positive is at the first position
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels