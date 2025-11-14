# models/adadro_model.py
import torch
import torch.nn as nn
from .backbone import create_backbone
from .moco import MoCo
from config.base_config import ModelConfig

class AdaDROModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(AdaDROModel, self).__init__()
        
        self.config = config
        
        # Create shared encoder
        if config.arch == 'simple':
            self.encoder = create_backbone('simple', config.feature_dim, pretrained=False)
        else:
            self.encoder = create_backbone(config.arch, config.feature_dim, pretrained=False)
        
        # Classification head (directly connected to encoder)
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)
        
        # MoCo (shared encoder, add projector)
        self.moco = MoCo(
            encoder=self.encoder,
            feature_dim=config.feature_dim,  # Explicitly pass feature dimension
            dim=config.moco_dim,
            K=config.moco_k,
            m=config.moco_m,
            T=config.moco_t,
            mlp=config.use_mlp
        )
        
    def forward(self, x):
        """Classification forward pass"""
        features = self.encoder.get_features(x)  # [B, feature_dim]
        logits = self.classifier(features)        # [B, num_classes]
        return features, logits
    
    def moco_forward(self, im_q, im_k):
        """Contrastive learning forward pass"""
        return self.moco(im_q, im_k)
    
    def get_classifier_weights(self):
        return self.classifier.weight
    
    def extract_features(self, x):
        with torch.no_grad():
            features = self.encoder.get_features(x)
        return features