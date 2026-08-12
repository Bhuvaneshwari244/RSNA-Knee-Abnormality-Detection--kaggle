"""
Model architectures for knee abnormality detection
"""
import torch
import torch.nn as nn
import timm


class KneeClassifier(nn.Module):
    """
    Image-only classifier for knee abnormalities
    Uses pretrained CNN backbone from timm library
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 12,
        pretrained: bool = True
    ):
        """
        Args:
            model_name: Name of model from timm library
            num_classes: Number of target classes (12 abnormalities)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=""  # Remove global pooling
        )
        
        # Get number of features from backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            num_features = features.shape[1]
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Logits for each class (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)  # (B, C, H, W)
        
        # Pool
        features = self.pool(features)  # (B, C, 1, 1)
        features = features.flatten(1)  # (B, C)
        
        # Classify
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits


class MultiModalKneeClassifier(nn.Module):
    """
    Multi-modal classifier combining images and radiology reports
    """
    
    def __init__(
        self,
        image_model_name: str = "efficientnet_b0",
        text_model_name: str = "distilbert-base-uncased",
        num_classes: int = 12,
        pretrained: bool = True
    ):
        """
        Args:
            image_model_name: CNN model name from timm
            text_model_name: Transformer model name from HuggingFace
            num_classes: Number of target classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Image encoder
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        
        # Get image feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            img_features = self.image_encoder(dummy_input)
            self.img_feature_dim = img_features.shape[1]
        
        # Text encoder (placeholder - would use transformers library)
        # For now, just using a simple embedding
        self.text_feature_dim = 768
        
        # Fusion layer
        fusion_dim = self.img_feature_dim + self.text_feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, images, text_embeddings=None):
        """
        Forward pass
        
        Args:
            images: Image tensor (B, 3, H, W)
            text_embeddings: Text embeddings (B, text_dim)
            
        Returns:
            Logits (B, num_classes)
        """
        # Image features
        img_features = self.image_encoder(images)  # (B, img_dim)
        
        # If no text, use image only
        if text_embeddings is None:
            text_embeddings = torch.zeros(
                img_features.size(0),
                self.text_feature_dim,
                device=img_features.device
            )
        
        # Concatenate features
        combined = torch.cat([img_features, text_embeddings], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 12,
    pretrained: bool = True,
    multimodal: bool = False
):
    """
    Factory function to create model
    
    Args:
        model_name: Model architecture name
        num_classes: Number of classes
        pretrained: Use pretrained weights
        multimodal: Use multimodal architecture
        
    Returns:
        Model instance
    """
    if multimodal:
        return MultiModalKneeClassifier(
            image_model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
    else:
        return KneeClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
