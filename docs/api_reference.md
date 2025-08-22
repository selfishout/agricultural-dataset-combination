# 📚 API Reference

Complete API documentation for the Agricultural Dataset Combination project. This document provides detailed information about all classes, functions, and modules.

## 📖 **Table of Contents**

- [Core Modules](#core-modules)
- [Dataset Combination](#dataset-combination)
- [Preprocessing](#preprocessing)
- [Visualization](#visualization)
- [Utilities](#utilities)
- [Configuration](#configuration)
- [Sample Project](#sample-project)

## 🏗️ **Core Modules**

### **src.dataset_combiner**

Main orchestration module for combining agricultural datasets.

#### **DatasetCombiner**

```python
class DatasetCombiner:
    """Main class for orchestrating dataset combination process."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the dataset combiner.
        
        Args:
            config: Configuration dictionary containing processing parameters
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If required directories don't exist
        """
    
    def combine_all_datasets(self) -> Dict[str, Any]:
        """
        Combine all available datasets into a unified format.
        
        Returns:
            Dictionary containing processing results and statistics
            
        Raises:
            RuntimeError: If dataset combination fails
            MemoryError: If insufficient memory for processing
        """
    
    def combine_specific_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Combine a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to combine
            
        Returns:
            Dictionary containing dataset-specific results
        """
    
    def validate_combination(self) -> Dict[str, Any]:
        """
        Validate the combined dataset for quality and consistency.
        
        Returns:
            Dictionary containing validation results
        """
```

#### **Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `combine_all_datasets()` | Combine all available datasets | None | Processing results |
| `combine_specific_dataset(name)` | Combine specific dataset | `dataset_name: str` | Dataset results |
| `validate_combination()` | Validate combined dataset | None | Validation results |
| `get_statistics()` | Get dataset statistics | None | Statistics dict |
| `cleanup_intermediate()` | Remove temporary files | None | Cleanup status |

### **src.dataset_loader**

Dataset loading utilities for different agricultural datasets.

#### **PhenoBenchLoader**

```python
class PhenoBenchLoader:
    """Loader for PhenoBench dataset."""
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]) -> None:
        """
        Initialize PhenoBench loader.
        
        Args:
            dataset_path: Path to PhenoBench dataset
            config: Configuration parameters
        """
    
    def load_images(self) -> List[str]:
        """Load all image paths from the dataset."""
        
    def load_annotations(self) -> List[str]:
        """Load all annotation paths from the dataset."""
        
    def validate_pairs(self) -> List[Tuple[str, str]]:
        """Validate image-annotation pairs."""
```

#### **CapsicumLoader**

```python
class CapsicumLoader:
    """Loader for Capsicum Annuum dataset."""
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]) -> None:
        """
        Initialize Capsicum loader.
        
        Args:
            dataset_path: Path to Capsicum dataset
            config: Configuration parameters
        """
    
    def extract_archive(self) -> str:
        """Extract compressed dataset archive."""
        
    def load_synthetic_data(self) -> List[str]:
        """Load synthetic pepper plant images."""
        
    def load_empirical_data(self) -> List[str]:
        """Load empirical pepper plant images."""
```

#### **VineyardLoader**

```python
class VineyardLoader:
    """Loader for Vineyard dataset."""
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]) -> None:
        """
        Initialize Vineyard loader.
        
        Args:
            dataset_path: Path to Vineyard dataset
            config: Configuration parameters
        """
    
    def load_canopy_images(self) -> List[str]:
        """Load vineyard canopy images."""
        
    def load_growth_labels(self) -> Dict[str, str]:
        """Load growth stage labels."""
```

## 🔄 **Dataset Combination**

### **Combination Workflow**

```python
# Complete workflow example
from src.dataset_combiner import DatasetCombiner
from config.dataset_config import load_config

# Load configuration
config = load_config('config/dataset_config.yaml')

# Create combiner
combiner = DatasetCombiner(config)

# Combine all datasets
results = combiner.combine_all_datasets()

# Validate results
validation = combiner.validate_combination()

# Get statistics
stats = combiner.get_statistics()
```

### **Configuration Parameters**

```yaml
# config/dataset_config.yaml
storage:
  output_dir: "data/final/combined_dataset"
  intermediate_dir: "data/intermediate"
  backup_original: true

processing:
  target_size: [512, 512]
  image_format: "PNG"
  augmentation: true
  quality_check: true
  batch_size: 8

splits:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1

datasets:
  phenobench:
    enabled: true
    priority: 1
  capsicum:
    enabled: true
    priority: 2
  vineyard:
    enabled: true
    priority: 3
```

## 🎨 **Preprocessing**

### **src.preprocessing**

Image and annotation preprocessing pipeline.

#### **ImagePreprocessor**

```python
class ImagePreprocessor:
    """Handles image preprocessing and transformation."""
    
    def __init__(self, target_size: Tuple[int, int], 
                 format: str = "PNG", 
                 augmentation: bool = True) -> None:
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image dimensions (width, height)
            format: Output image format
            augmentation: Enable data augmentation
        """
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image array
            
        Returns:
            Resized image array
        """
    
    def convert_format(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to specified format.
        
        Args:
            image: Input image array
            
        Returns:
            Converted image array
        """
    
    def apply_augmentation(self, image: np.ndarray, 
                          mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to image and mask.
        
        Args:
            image: Input image array
            mask: Input mask array (optional)
            
        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
```

#### **AnnotationProcessor**

```python
class AnnotationProcessor:
    """Handles annotation preprocessing and conversion."""
    
    def __init__(self, target_size: Tuple[int, int], 
                 format: str = "PNG") -> None:
        """
        Initialize annotation processor.
        
        Args:
            target_size: Target annotation dimensions
            format: Output annotation format
        """
    
    def resize_annotation(self, annotation: np.ndarray) -> np.ndarray:
        """
        Resize annotation to target dimensions.
        
        Args:
            annotation: Input annotation array
            
        Returns:
            Resized annotation array
        """
    
    def convert_format(self, annotation: np.ndarray) -> np.ndarray:
        """
        Convert annotation to specified format.
        
        Args:
            annotation: Input annotation array
            
        Returns:
            Converted annotation array
        """
    
    def validate_annotation(self, annotation: np.ndarray) -> bool:
        """
        Validate annotation for consistency.
        
        Args:
            annotation: Input annotation array
            
        Returns:
            True if annotation is valid
        """
```

### **Preprocessing Pipeline**

```python
# Complete preprocessing example
from src.preprocessing import ImagePreprocessor, AnnotationProcessor

# Initialize processors
img_processor = ImagePreprocessor(target_size=(512, 512), format="PNG")
ann_processor = AnnotationProcessor(target_size=(512, 512), format="PNG")

# Process image
processed_image = img_processor.resize_image(image)
processed_image = img_processor.convert_format(processed_image)

# Process annotation
processed_annotation = ann_processor.resize_annotation(annotation)
processed_annotation = ann_processor.convert_format(processed_annotation)

# Apply augmentation
aug_image, aug_annotation = img_processor.apply_augmentation(
    processed_image, processed_annotation
)
```

## 📊 **Visualization**

### **src.visualization**

Dataset visualization and reporting tools.

#### **DatasetVisualizer**

```python
class DatasetVisualizer:
    """Creates visualizations and reports for datasets."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize dataset visualizer.
        
        Args:
            config: Configuration parameters
        """
    
    def create_dataset_overview(self, stats: Dict[str, Any]) -> str:
        """
        Create dataset overview visualization.
        
        Args:
            stats: Dataset statistics
            
        Returns:
            Path to saved visualization
        """
    
    def create_sample_showcase(self, dataset_path: str, 
                              num_samples: int = 16) -> str:
        """
        Create sample image showcase.
        
        Args:
            dataset_path: Path to dataset
            num_samples: Number of samples to showcase
            
        Returns:
            Path to saved visualization
        """
    
    def create_quality_report(self, quality_data: Dict[str, Any]) -> str:
        """
        Create quality control report.
        
        Args:
            quality_data: Quality control results
            
        Returns:
            Path to saved report
        """
    
    def create_interactive_dashboard(self, stats: Dict[str, Any], 
                                   results: Dict[str, Any], 
                                   quality: Dict[str, Any]) -> str:
        """
        Create interactive dashboard.
        
        Args:
            stats: Dataset statistics
            results: Processing results
            quality: Quality control data
            
        Returns:
            Path to saved dashboard
        """
```

### **Visualization Examples**

```python
# Create comprehensive visualizations
from src.visualization import DatasetVisualizer

# Initialize visualizer
visualizer = DatasetVisualizer(config)

# Create overview
overview_path = visualizer.create_dataset_overview(stats)

# Create sample showcase
showcase_path = visualizer.create_sample_showcase(
    dataset_path, num_samples=25
)

# Create quality report
quality_path = visualizer.create_quality_report(quality_data)

# Create interactive dashboard
dashboard_path = visualizer.create_interactive_dashboard(
    stats, results, quality
)
```

## 🛠️ **Utilities**

### **src.utils**

General utility functions and helpers.

#### **File Utilities**

```python
def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path to ensure
    """

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """

def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """

def find_duplicates(file_paths: List[str]) -> List[List[str]]:
    """
    Find duplicate files based on MD5 hash.
    
    Args:
        file_paths: List of file paths to check
        
    Returns:
        List of duplicate file groups
    """
```

#### **Image Utilities**

```python
def is_valid_image(file_path: str) -> bool:
    """
    Check if file is a valid image.
    
    Args:
        file_path: Path to image file
        
    Returns:
        True if file is valid image
    """

def get_image_info(file_path: str) -> Dict[str, Any]:
    """
    Get image information.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Dictionary containing image information
    """

def validate_image_annotation_pair(image_path: str, 
                                 annotation_path: str) -> bool:
    """
    Validate image-annotation pair.
    
    Args:
        image_path: Path to image file
        annotation_path: Path to annotation file
        
    Returns:
        True if pair is valid
    """
```

#### **Progress Tracking**

```python
def create_progress_bar(total: int, description: str = "") -> tqdm:
    """
    Create progress bar for long-running operations.
    
    Args:
        total: Total number of items
        description: Description of operation
        
    Returns:
        tqdm progress bar object
    """

def log_progress(message: str, level: str = "INFO") -> None:
    """
    Log progress message.
    
    Args:
        message: Progress message
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
```

## ⚙️ **Configuration**

### **Configuration Management**

```python
# config/dataset_config.py
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
```

### **Configuration Schema**

```yaml
# Configuration schema
storage:
  output_dir: string          # Required: Output directory path
  intermediate_dir: string    # Required: Intermediate files directory
  backup_original: boolean    # Optional: Backup original datasets

processing:
  target_size: [integer, integer]  # Required: [width, height]
  image_format: string             # Required: Output image format
  augmentation: boolean            # Optional: Enable augmentation
  quality_check: boolean          # Optional: Enable quality checks
  batch_size: integer             # Optional: Processing batch size

splits:
  train_ratio: float              # Required: Training split ratio
  val_ratio: float                # Required: Validation split ratio
  test_ratio: float               # Required: Test split ratio

datasets:
  dataset_name:
    enabled: boolean              # Required: Enable/disable dataset
    priority: integer             # Optional: Processing priority
    custom_config: object         # Optional: Dataset-specific config
```

## 🎨 **Sample Project**

### **sample_segmentation_project**

Complete image segmentation project using the combined dataset.

#### **Configuration**

```python
# sample_segmentation_project/config.py
class Config:
    """Configuration for segmentation project."""
    
    # Dataset paths
    DATASET_PATH = "/path/to/combined/dataset"
    TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    VAL_PATH = os.path.join(DATASET_PATH, "val")
    TEST_PATH = os.path.join(DATASET_PATH, "test")
    
    # Model parameters
    INPUT_SIZE = (512, 512)
    NUM_CHANNELS = 3
    NUM_CLASSES = 1
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Model architecture
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DECODER_CHANNELS = [256, 128, 64, 32]
    
    # Data augmentation
    AUGMENTATION_PROB = 0.5
    ROTATION_LIMIT = 30
    BRIGHTNESS_LIMIT = 0.2
    
    # Loss function
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Directories
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
```

#### **Dataset Loading**

```python
# sample_segmentation_project/dataset.py
class AgriculturalSegmentationDataset(Dataset):
    """Dataset for agricultural image segmentation."""
    
    def __init__(self, images_dir: str, masks_dir: str, 
                 transform: Optional[Callable] = None) -> None:
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            transform: Optional transformations
        """
    
    def __len__(self) -> int:
        """Return number of samples."""
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and mask tensors
        """
```

#### **Model Architecture**

```python
# sample_segmentation_project/model.py
class UNet(nn.Module):
    """U-Net architecture for image segmentation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
```

#### **Training Pipeline**

```python
# sample_segmentation_project/train_segmentation.py
class Trainer:
    """Training pipeline for segmentation model."""
    
    def __init__(self, config: Config) -> None:
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
    
    def train(self) -> None:
        """Run complete training pipeline."""
```

## 📊 **API Usage Examples**

### **Complete Workflow Example**

```python
# Complete dataset combination and training workflow
from src.dataset_combiner import DatasetCombiner
from src.visualization import DatasetVisualizer
from config.dataset_config import load_config
from sample_segmentation_project.train_segmentation import Trainer
from sample_segmentation_project.config import Config

# 1. Load configuration
config = load_config('config/dataset_config.yaml')

# 2. Combine datasets
combiner = DatasetCombiner(config)
results = combiner.combine_all_datasets()

# 3. Create visualizations
visualizer = DatasetVisualizer(config)
overview = visualizer.create_dataset_overview(results['statistics'])
showcase = visualizer.create_sample_showcase(config['storage']['output_dir'])

# 4. Train segmentation model
trainer = Trainer(Config())
trainer.train()

print("✅ Complete workflow executed successfully!")
```

### **Custom Dataset Integration**

```python
# Example of adding custom dataset support
class CustomDatasetLoader:
    """Custom dataset loader example."""
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]) -> None:
        self.dataset_path = dataset_path
        self.config = config
    
    def load_data(self) -> List[str]:
        """Load custom dataset data."""
        # Implement custom loading logic
        pass
    
    def validate_data(self) -> bool:
        """Validate custom dataset."""
        # Implement custom validation
        pass

# Register custom loader
custom_loader = CustomDatasetLoader("/path/to/custom/dataset", config)
```

### **Advanced Preprocessing**

```python
# Custom preprocessing pipeline
from src.preprocessing import ImagePreprocessor

class CustomImagePreprocessor(ImagePreprocessor):
    """Custom image preprocessor with additional features."""
    
    def apply_custom_transforms(self, image: np.ndarray) -> np.ndarray:
        """Apply custom transformations."""
        # Implement custom transformations
        return image
    
    def batch_process(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Process multiple images in batch."""
        return [self.apply_custom_transforms(img) for img in images]

# Use custom preprocessor
custom_processor = CustomImagePreprocessor(target_size=(1024, 1024))
processed_images = custom_processor.batch_process(image_list)
```

## 🔍 **Error Handling**

### **Common Exceptions**

```python
# Exception handling examples
try:
    combiner = DatasetCombiner(config)
    results = combiner.combine_all_datasets()
except ValueError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
except FileNotFoundError as e:
    print(f"File not found: {e}")
    # Handle missing files
except MemoryError as e:
    print(f"Insufficient memory: {e}")
    # Handle memory issues
except RuntimeError as e:
    print(f"Runtime error: {e}")
    # Handle processing errors
```

### **Validation and Error Checking**

```python
# Validate before processing
if not combiner.validate_config():
    print("Configuration validation failed")
    exit(1)

if not combiner.check_dependencies():
    print("Dependency check failed")
    exit(1)

# Safe processing with error handling
try:
    results = combiner.combine_all_datasets()
    print("Dataset combination completed successfully")
except Exception as e:
    print(f"Dataset combination failed: {e}")
    # Implement fallback or recovery
```

---

## 🎯 **API Summary**

### **Core Classes**

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `DatasetCombiner` | Main orchestration | `combine_all_datasets()`, `validate_combination()` |
| `ImagePreprocessor` | Image processing | `resize_image()`, `convert_format()`, `apply_augmentation()` |
| `AnnotationProcessor` | Annotation processing | `resize_annotation()`, `convert_format()`, `validate_annotation()` |
| `DatasetVisualizer` | Visualization | `create_dataset_overview()`, `create_interactive_dashboard()` |

### **Key Functions**

| Function | Purpose | Returns |
|----------|---------|---------|
| `load_config()` | Load configuration | Configuration dictionary |
| `ensure_directory()` | Create directories | None |
| `calculate_md5()` | Calculate file hash | MD5 hash string |
| `is_valid_image()` | Validate image file | Boolean |
| `create_progress_bar()` | Create progress bar | tqdm object |

### **Configuration Options**

| Section | Key Parameters | Description |
|---------|----------------|-------------|
| `storage` | `output_dir`, `intermediate_dir` | File system paths |
| `processing` | `target_size`, `image_format`, `augmentation` | Processing parameters |
| `splits` | `train_ratio`, `val_ratio`, `test_ratio` | Dataset splitting |
| `datasets` | `enabled`, `priority`, `custom_config` | Dataset-specific settings |

---

**This API reference covers all major components of the Agricultural Dataset Combination project. For detailed examples and advanced usage, refer to the individual module documentation and source code.** 🚀

---

<div align="center">

**Need More Details?** Check our [Complete Documentation](README.md) or [Source Code](https://github.com/selfishout/agricultural-dataset-combination)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/selfishout)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-torabi)

</div>
