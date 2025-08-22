"""
Configuration file for the Sample Image Segmentation Project.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the segmentation project."""
    
    # Dataset paths
    DATASET_ROOT = "/Volumes/Rapid/Agriculture Dataset/Combined_datasets"
    TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
    TRAIN_ANNOTATIONS = os.path.join(DATASET_ROOT, "train", "annotations")
    VAL_IMAGES = os.path.join(DATASET_ROOT, "val", "images")
    VAL_ANNOTATIONS = os.path.join(DATASET_ROOT, "val", "annotations")
    TEST_IMAGES = os.path.join(DATASET_ROOT, "test", "images")
    TEST_ANNOTATIONS = os.path.join(DATASET_ROOT, "test", "annotations")
    
    # Model parameters
    INPUT_SIZE = (512, 512)
    NUM_CHANNELS = 3  # RGB
    NUM_CLASSES = 1   # Binary segmentation
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Model architecture
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DECODER_CHANNELS = [256, 128, 64, 64]
    
    # Data augmentation
    AUGMENTATION_PROB = 0.5
    ROTATION_LIMIT = 30
    SHIFT_LIMIT = 0.1
    SCALE_LIMIT = 0.2
    
    # Loss function weights
    DICE_LOSS_WEIGHT = 0.5
    BCE_LOSS_WEIGHT = 0.5
    
    # Training settings
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Checkpoint and logging
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    # Evaluation
    EVAL_BATCH_SIZE = 16
    SAVE_PREDICTIONS = True
    
    # Visualization
    NUM_SAMPLES_TO_VISUALIZE = 10
    SAVE_VISUALIZATIONS = True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for dir_path in [cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.RESULTS_DIR]:
            Path(dir_path).mkdir(exist_ok=True)
    
    @classmethod
    def validate_paths(cls):
        """Validate that all dataset paths exist."""
        paths_to_check = [
            cls.TRAIN_IMAGES, cls.TRAIN_ANNOTATIONS,
            cls.VAL_IMAGES, cls.VAL_ANNOTATIONS,
            cls.TEST_IMAGES, cls.TEST_ANNOTATIONS
        ]
        
        missing_paths = []
        for path in paths_to_check:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            raise FileNotFoundError(f"Missing dataset paths: {missing_paths}")
        
        print("✅ All dataset paths validated successfully!")
        return True

# Create configuration instance
config = Config()
