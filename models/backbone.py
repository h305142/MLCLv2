import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class ResNetBackbone(nn.Module):
    def __init__(self, arch: str = 'resnet18', num_classes: int = 1000, pretrained: bool = False):
        super(ResNetBackbone, self).__init__()
        
        if arch == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif arch == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif arch == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before final classification layer"""
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        return features

class SimpleEncoder(nn.Module):
    def __init__(self, num_classes: int = 10, feature_dim: int = 128):
        super(SimpleEncoder, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(128, num_classes)
        self.feature_dim = 128
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return logits
    
    def get_features(self, x):
        """Extract features before final classification layer"""
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features

def create_backbone(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Create backbone network
    
    Args:
        arch: Architecture name ('resnet18', 'resnet34', 'resnet50', 'simple')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Backbone network module
    """
    if 'resnet' in arch:
        return ResNetBackbone(arch, num_classes, pretrained)
    elif arch == 'simple':
        return SimpleEncoder(num_classes)
    else:
        raise ValueError(f"Unsupported backbone architecture: {arch}")