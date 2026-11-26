"""
Deep Learning Face Feature Extractor.
Uses Convolutional Neural Networks (Transfer Learning with MobileNetV2/ResNet18 or Custom CNN)
to map face crops into dense, normalized embedding vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Union
import config

class CustomFaceCNN(nn.Module):
    """
    Custom 5-layer Convolutional Neural Network for Face Representation.
    Built for academic demonstration of training from scratch.
    Weights are deterministically seeded.
    """
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112x112
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56x56
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 28x28
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14x14
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


class MobileNetV2FaceEncoder(nn.Module):
    """
    Transfer learning face encoder based on MobileNetV2 backbone.
    High accuracy, lightweight, and fast inference on CPU.
    Projection head is deterministically initialized.
    """
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Deterministic seed for projection head
        torch.manual_seed(42)
        self.head = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


class ResNet18FaceEncoder(nn.Module):
    """
    Transfer learning face encoder based on ResNet18 backbone.
    Residual connections provide strong representational power.
    Uses the native 512-dimensional pooled feature representations directly.
    """
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2, dim=1)


class FaceFeatureExtractor:
    """
    Unified interface to extract standardized face embedding vectors.
    """
    def __init__(self, backbone: str = config.DEFAULT_BACKBONE, 
                 embedding_dim: int = config.EMBEDDING_DIM,
                 device: Optional[str] = None):
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Instantiate requested backbone
        if backbone == "mobilenet_v2":
            self.model = MobileNetV2FaceEncoder(embedding_dim=embedding_dim, pretrained=True)
        elif backbone == "resnet18":
            self.model = ResNet18FaceEncoder(embedding_dim=embedding_dim, pretrained=True)
        elif backbone == "custom_cnn":
            self.model = CustomFaceCNN(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from 'mobilenet_v2', 'resnet18', 'custom_cnn'")

        self.model.to(self.device)
        self.model.eval()

        # Input transformation matching PyTorch vision models
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Scales [0, 255] to [0.0, 1.0]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts an L2-normalized 1D feature vector from a face crop (RGB format).
        Returns a numpy array of shape (embedding_dim,).
        """
        if face_rgb is None or face_rgb.size == 0:
            raise ValueError("Input face image is empty.")

        # Ensure correct data type
        if face_rgb.dtype != np.uint8:
            face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)

        # Transform and add batch dimension
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)
            # Transfer to CPU numpy array
            embedding_np = embedding.squeeze(0).cpu().numpy()

        # Re-ensure exact unit norm
        norm = np.linalg.norm(embedding_np)
        if norm > 1e-6:
            embedding_np = embedding_np / norm

        return embedding_np.astype(np.float32)

    @staticmethod
    def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray, metric: str = "cosine") -> float:
        """
        Computes the similarity score between two normalized embeddings.
        Returns a score in range [0.0, 1.0], where higher indicates greater similarity.
        """
        a = np.asarray(embedding_a, dtype=np.float32).flatten()
        b = np.asarray(embedding_b, dtype=np.float32).flatten()

        if metric == "cosine":
            # For unit vectors, cosine similarity = dot product
            dot = float(np.dot(a, b))
            # Calibrated biometric confidence: maps raw dot product in [0.70, 1.00] to intuitive range [0.0, 1.0]
            calibrated = (dot - 0.70) / 0.30
            return float(np.clip(calibrated, 0.0, 1.0))
        elif metric == "euclidean":
            # Euclidean distance
            dist = float(np.linalg.norm(a - b))
            # Convert distance to similarity score: exp(-dist)
            score = float(np.exp(-dist))
            return score
        else:
            raise ValueError(f"Unsupported metric: {metric}")
