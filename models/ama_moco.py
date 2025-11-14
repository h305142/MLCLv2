import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist




class AMA_MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=4096, m=0.999, T=0.07, T_local=0.07,
                 scales=[10, 30, 100], weight_momentum=0.1, cluster_update_freq=500):
        super().__init__()
        
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.T_local = T_local
        self.scales = scales
        self.weight_momentum = weight_momentum
        self.cluster_update_freq = cluster_update_freq
        
        # Encoder
        self.encoder_q = encoder
        self.encoder_k = self._build_encoder_k(encoder)
        
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Global queue
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Multi-scale components
        for scale in scales:
            scale_name = f'scale_{scale}'
            
            centers = F.normalize(torch.randn(scale, dim), dim=1)
            self.register_buffer(f'{scale_name}_centers', centers)
            
            local_queue = F.normalize(torch.randn(dim, K), dim=0)
            self.register_buffer(f'{scale_name}_local_queue', local_queue)
            self.register_buffer(f'{scale_name}_local_ptr', torch.zeros(1, dtype=torch.long))
            
            self.register_buffer(f'{scale_name}_queue_clusters', torch.zeros(K, dtype=torch.long))
            self.register_buffer(f'{scale_name}_log_weights', torch.zeros(scale))
        
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))
    
    def _build_encoder_k(self, encoder_q):
        import copy
        return copy.deepcopy(encoder_q)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        self.queue_ptr[0] = (ptr + batch_size) % self.K
    
    @torch.no_grad()
    def _dequeue_and_enqueue_local(self, keys, clusters, scale_name):
        batch_size = keys.shape[0]
        
        local_queue = getattr(self, f'{scale_name}_local_queue')
        local_ptr = getattr(self, f'{scale_name}_local_ptr')
        queue_clusters = getattr(self, f'{scale_name}_queue_clusters')
        
        ptr = int(local_ptr)
        
        if ptr + batch_size <= self.K:
            local_queue[:, ptr:ptr + batch_size] = keys.T
            queue_clusters[ptr:ptr + batch_size] = clusters
        else:
            remaining = self.K - ptr
            local_queue[:, ptr:] = keys[:remaining].T
            queue_clusters[ptr:] = clusters[:remaining]
            local_queue[:, :batch_size - remaining] = keys[remaining:].T
            queue_clusters[:batch_size - remaining] = clusters[remaining:]
        
        local_ptr[0] = (ptr + batch_size) % self.K
    
    @torch.no_grad()
    def _update_clusters(self, features, scale_name, n_iter=10):
        centers = getattr(self, f'{scale_name}_centers')
        
        for _ in range(n_iter):
            similarities = torch.mm(features, centers.T)
            assignments = similarities.argmax(dim=1)
            
            new_centers = []
            for k in range(centers.shape[0]):
                mask = (assignments == k)
                if mask.sum() > 0:
                    new_centers.append(features[mask].mean(dim=0))
                else:
                    new_centers.append(centers[k])
            
            centers = torch.stack(new_centers)
            centers = F.normalize(centers, dim=1)
        
        centers_buffer = getattr(self, f'{scale_name}_centers')
        centers_buffer.copy_(centers)
    
    def _get_cluster_assignments(self, features, scale_name):
        centers = getattr(self, f'{scale_name}_centers')
        similarities = torch.mm(features.detach(), centers.T)
        return similarities.argmax(dim=1)
    
    def _get_cluster_weights(self, scale_name):
        log_weights = getattr(self, f'{scale_name}_log_weights')
        return F.softmax(log_weights, dim=0)
    
    @torch.no_grad()
    def _update_cluster_weights(self, scale_name, cluster_ids, per_sample_losses):
        num_clusters = int(scale_name.split('_')[1])
        
        cluster_losses = torch.zeros(num_clusters, device=per_sample_losses.device)
        cluster_counts = torch.zeros(num_clusters, device=per_sample_losses.device)
        
        cluster_losses.scatter_add_(0, cluster_ids, per_sample_losses)
        cluster_counts.scatter_add_(0, cluster_ids, torch.ones_like(per_sample_losses))
        
        valid_mask = cluster_counts > 0
        cluster_losses[valid_mask] /= cluster_counts[valid_mask]
        
        # Direct in-place update (safe under no_grad)
        log_weights = getattr(self, f'{scale_name}_log_weights')
        log_weights[valid_mask] = (
            (1 - self.weight_momentum) * log_weights[valid_mask] + 
            self.weight_momentum * cluster_losses[valid_mask]
        )
    
    def forward(self, im_q, im_k):
        q = F.normalize(self.encoder_q(im_q), dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(im_k), dim=1)
        
        # Global contrastive loss
        l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(1)
        l_neg = torch.einsum('bd,dk->bk', q, self.queue.detach())
        
        logits_global = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        
        loss_global = F.cross_entropy(logits_global, labels)
        
        # Multi-scale local contrastive loss
        loss_local = 0
        
        for scale in self.scales:
            scale_name = f'scale_{scale}'
            
            with torch.no_grad():
                local_queue = getattr(self, f'{scale_name}_local_queue')
                queue_clusters = getattr(self, f'{scale_name}_queue_clusters')
                q_clusters = self._get_cluster_assignments(q, scale_name)
            
            cluster_mask = (q_clusters.unsqueeze(1) == queue_clusters.unsqueeze(0))
            l_neg_all = torch.einsum('bd,dk->bk', q, local_queue)
            
            has_neighbors = cluster_mask.sum(dim=1) > 0
            
            # Simplified empty cluster handling
            l_neg_masked = torch.where(
                cluster_mask,
                l_neg_all,
                torch.full_like(l_neg_all, -1e10)
            )
            
            l_neg_local = torch.where(
                has_neighbors.unsqueeze(1),
                l_neg_masked,
                l_neg_all
            )
            
            # Local loss: negative samples only
            logits_local = l_neg_local / self.T_local
            per_sample_losses = -torch.logsumexp(logits_local, dim=1)
            
            weights = self._get_cluster_weights(scale_name)
            sample_weights = weights[q_clusters]
            
            loss_local += (per_sample_losses * sample_weights).mean()
            
            if self.training:
                self._update_cluster_weights(scale_name, q_clusters, per_sample_losses.detach())
        
        loss_local /= len(self.scales)
        loss = loss_global + loss_local
        
        self._cached_k = k.detach()
        
        return loss
    
    @torch.no_grad()
    def sync_queues_ddp(self):
        """Synchronize queues across all processes in DDP mode"""
        if not dist.is_initialized():
            return  # Not in DDP mode, return directly
        
        # Synchronize global queue
        dist.broadcast(self.queue, src=0)
        dist.broadcast(self.queue_ptr, src=0)
        
        # Synchronize multi-scale queues
        for scale in self.scales:
            scale_name = f'scale_{scale}'
            local_queue = getattr(self, f'{scale_name}_local_queue')
            local_ptr = getattr(self, f'{scale_name}_local_ptr')
            queue_clusters = getattr(self, f'{scale_name}_queue_clusters')
            
            dist.broadcast(local_queue, src=0)
            dist.broadcast(local_ptr, src=0)
            dist.broadcast(queue_clusters, src=0)
    
    @torch.no_grad()
    def update_queues(self):
        if not hasattr(self, '_cached_k'):
            return
        
        k = self._cached_k
        
        # In DDP mode, gather k from all processes
        if dist.is_initialized():
            world_size = dist.get_world_size()
            k_list = [torch.zeros_like(k) for _ in range(world_size)]
            dist.all_gather(k_list, k)
            k = torch.cat(k_list, dim=0)  # Concatenate features from all GPUs
        
        self._dequeue_and_enqueue(k)
        
        for scale in self.scales:
            scale_name = f'scale_{scale}'
            k_clusters = self._get_cluster_assignments(k, scale_name)
            self._dequeue_and_enqueue_local(k, k_clusters, scale_name)
        
        self.step_counter += 1
        if self.step_counter % self.cluster_update_freq == 0:
            for scale in self.scales:
                scale_name = f'scale_{scale}'
                queue_features = getattr(self, f'{scale_name}_local_queue').T
                self._update_clusters(queue_features, scale_name)
        
        # Synchronize queues to all processes
        if dist.is_initialized():
            self.sync_queues_ddp()
        
        delattr(self, '_cached_k')