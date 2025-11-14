import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """ResNet encoder that outputs normalized feature vectors"""
    
    def __init__(self, base_model='resnet50', dim=128):
        super().__init__()
        
        # Load pretrained ResNet (remove final FC layer)
        if base_model == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        elif base_model == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        else:
            raise ValueError(f"Unsupported model: {base_model}")
        
        # Remove final fully connected layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add projection head (2048 -> dim)
        self.projector = nn.Sequential(
            nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
            nn.ReLU(inplace=True),
            nn.Linear(resnet.fc.in_features, dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, dim] unnormalized features
        """
        h = self.encoder(x)  # [B, 2048, 1, 1]
        h = h.flatten(1)     # [B, 2048]
        z = self.projector(h)  # [B, dim]
        return z