"""
Configuration settings for the Advanced Facial Recognition System.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
FACES_DIR = DATABASE_DIR / "faces"
USERS_FILE = DATABASE_DIR / "enrolled_users.json"
EVALUATION_DIR = BASE_DIR / "evaluation_results"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# Ensure directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Face Detection Settings
FACE_DETECTION_CONFIDENCE = 0.5
FACE_MIN_SIZE = (50, 50)
FACE_PADDING_RATIO = 0.2  # 20% margin around detected face box
FACE_INPUT_SIZE = (224, 224)

# Deep Learning Feature Extractor Settings
# Supported backbones: 'mobilenet_v2', 'resnet18', 'custom_cnn'
DEFAULT_BACKBONE = "mobilenet_v2"
EMBEDDING_DIM = 512

# Verification & Matching Settings
# Metric: 'cosine' or 'euclidean'
SIMILARITY_METRIC = "cosine"
# Recommended threshold for cosine similarity (ranges 0.0 to 1.0)
# Matches >= DEFAULT_VERIFICATION_THRESHOLD will be accepted
DEFAULT_VERIFICATION_THRESHOLD = 0.65

# Anti-Spoofing / Liveness Settings
ENABLE_LIVENESS_CHECK = True
LIVENESS_LAPLACIAN_THRESHOLD = 50.0  # Texture sharpness threshold to detect blurry screen/paper attacks

