import torch
import torch.nn as nn
import torch.nn.functional as F








class FastClustering(nn.Module):
    """Fast K-means clustering using cosine similarity"""
    
    def __init__(self, num_clusters, dim):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        
        # Cluster centroids [num_clusters, dim]
        self.register_buffer(
            'centroids', 
            F.normalize(torch.randn(num_clusters, dim), dim=1)
        )
    
    @torch.no_grad()
    def cluster(self, features, n_iter=10):
        """
        K-means clustering to update centroids
        
        Args:
            features: [N, D] features to cluster (already normalized)
            n_iter: int, number of iterations
        """
        features = F.normalize(features, dim=1)
        centroids = self.centroids.clone()
        
        for _ in range(n_iter):
            # Assign samples to nearest centroid
            similarity = torch.mm(features, centroids.t())  # [N, K]
            assignments = similarity.argmax(dim=1)  # [N]
            
            # Update centroids by averaging all samples assigned to each centroid
            new_centroids = []
            for k in range(self.num_clusters):
                mask = (assignments == k)
                if mask.sum() > 0:
                    new_centroids.append(features[mask].mean(dim=0))
                else:
                    # Keep centroid unchanged if no samples are assigned
                    new_centroids.append(centroids[k])
            
            centroids = torch.stack(new_centroids)
            centroids = F.normalize(centroids, dim=1)
        
        self.centroids.copy_(centroids)
    
    @torch.no_grad()
    def assign(self, features):
        """
        Assign samples to nearest cluster centroid
        
        Args:
            features: [B, D] normalized features
        
        Returns:
            assignments: [B] cluster labels
        """
        features = F.normalize(features, dim=1)
        similarity = torch.mm(features, self.centroids.t())  # [B, K]
        return similarity.argmax(dim=1)