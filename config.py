"""
Configuration file for RSNA Knee Abnormality Detection
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Create directories if they don't exist
for dir_path in [DATA_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Target labels (12 abnormalities)
TARGETS = [
    "ACL",
    "MCL", 
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

# Model hyperparameters
IMAGE_SIZE = 224  # Will be adjusted based on model choice
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

# Model architecture
MODEL_NAME = "efficientnet_b0"  # From timm library
PRETRAINED = True

# Training settings
DEVICE = "cuda"  # Will auto-fallback to CPU if CUDA unavailable
SEED = 42
MIXED_PRECISION = True  # Use AMP for faster training

# Cross-validation
N_FOLDS = 5
FOLD = 0  # Which fold to train on

# Inference
TEST_TIME_AUGMENTATION = False
ENSEMBLE = False
