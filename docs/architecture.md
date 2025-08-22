# 🏗️ Architecture Guide

Comprehensive architecture documentation for the Agricultural Dataset Combination project. This document explains the system design, components, and data flow.

## 📖 **Table of Contents**

- [System Overview](#system-overview)
- [Architecture Components](#architecture-components)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Performance Considerations](#performance-considerations)
- [Scalability](#scalability)
- [Security](#security)
- [Deployment](#deployment)

## 🎯 **System Overview**

### **High-Level Architecture**

The Agricultural Dataset Combination project follows a **modular, pipeline-based architecture** designed for:

- **Scalability**: Handle datasets of varying sizes
- **Flexibility**: Support different dataset formats and sources
- **Reliability**: Robust error handling and validation
- **Performance**: Efficient processing and memory management
- **Maintainability**: Clear separation of concerns

### **Architecture Diagram**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source        │    │   Processing    │    │   Output        │
│   Datasets      │───▶│   Pipeline      │───▶│   Combined      │
│                 │    │                 │    │   Dataset       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dataset       │    │   Quality       │    │   Sample        │
│   Loaders       │    │   Control       │    │   Project       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Core Principles**

1. **Separation of Concerns**: Each module has a single responsibility
2. **Configuration-Driven**: Behavior controlled through configuration files
3. **Pipeline Architecture**: Data flows through well-defined stages
4. **Error Resilience**: Graceful handling of failures and edge cases
5. **Extensibility**: Easy to add new dataset types and processing steps

## 🧩 **Architecture Components**

### **1. Core Modules (`src/`)**

#### **Dataset Combiner (`src/dataset_combiner.py`)**

**Purpose**: Main orchestration module that coordinates the entire dataset combination process.

**Responsibilities**:
- Manage the overall workflow
- Coordinate between different components
- Handle error recovery and logging
- Generate comprehensive reports

**Key Classes**:
```python
class DatasetCombiner:
    """Main orchestrator for dataset combination."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loaders = {}
        self.processors = {}
        self.validators = {}
    
    def combine_all_datasets(self) -> Dict[str, Any]:
        """Execute complete dataset combination workflow."""
        
    def combine_specific_dataset(self, name: str) -> Dict[str, Any]:
        """Combine a specific dataset."""
        
    def validate_combination(self) -> Dict[str, Any]:
        """Validate the combined dataset."""
```

**Design Patterns**:
- **Facade Pattern**: Provides simplified interface to complex subsystem
- **Strategy Pattern**: Different combination strategies for different datasets
- **Observer Pattern**: Progress tracking and logging

#### **Dataset Loaders (`src/dataset_loader.py`)**

**Purpose**: Handle loading and validation of different dataset formats.

**Responsibilities**:
- Load dataset-specific data structures
- Validate data integrity
- Convert to common format
- Handle dataset-specific quirks

**Key Classes**:
```python
class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_images(self) -> List[str]:
        """Load image paths."""
        
    @abstractmethod
    def load_annotations(self) -> List[str]:
        """Load annotation paths."""
        
    @abstractmethod
    def validate_pairs(self) -> List[Tuple[str, str]]:
        """Validate image-annotation pairs."""

class PhenoBenchLoader(BaseDatasetLoader):
    """Loader for PhenoBench dataset."""
    
class CapsicumLoader(BaseDatasetLoader):
    """Loader for Capsicum Annuum dataset."""
    
class VineyardLoader(BaseDatasetLoader):
    """Loader for Vineyard dataset."""
```

**Design Patterns**:
- **Template Method Pattern**: Common workflow with dataset-specific implementations
- **Factory Pattern**: Create appropriate loader based on dataset type
- **Adapter Pattern**: Convert dataset-specific formats to common interface

#### **Preprocessing (`src/preprocessing.py`)**

**Purpose**: Handle image and annotation preprocessing and transformation.

**Responsibilities**:
- Image resizing and format conversion
- Annotation processing and validation
- Data augmentation
- Quality control

**Key Classes**:
```python
class ImagePreprocessor:
    """Handle image preprocessing operations."""
    
    def __init__(self, target_size: Tuple[int, int], format: str):
        self.target_size = target_size
        self.format = format
        self.augmentations = self._create_augmentations()
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process single image."""
        
    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Process multiple images."""
        
    def apply_augmentation(self, image: np.ndarray, mask: np.ndarray = None):
        """Apply data augmentation."""

class AnnotationProcessor:
    """Handle annotation preprocessing operations."""
    
    def process_annotation(self, annotation: np.ndarray) -> np.ndarray:
        """Process single annotation."""
        
    def validate_annotation(self, annotation: np.ndarray) -> bool:
        """Validate annotation quality."""
```

**Design Patterns**:
- **Chain of Responsibility**: Processing steps can be chained
- **Decorator Pattern**: Add augmentation capabilities
- **Strategy Pattern**: Different processing strategies

#### **Visualization (`src/visualization.py`)**

**Purpose**: Generate visualizations and reports for datasets and processing results.

**Responsibilities**:
- Create dataset overview charts
- Generate sample image showcases
- Produce quality control reports
- Build interactive dashboards

**Key Classes**:
```python
class DatasetVisualizer:
    """Create comprehensive dataset visualizations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plot_style = self._setup_plot_style()
    
    def create_dataset_overview(self, stats: Dict[str, Any]) -> str:
        """Create dataset overview visualization."""
        
    def create_sample_showcase(self, dataset_path: str, num_samples: int) -> str:
        """Create sample image showcase."""
        
    def create_quality_report(self, quality_data: Dict[str, Any]) -> str:
        """Create quality control report."""
        
    def create_interactive_dashboard(self, stats: Dict[str, Any], 
                                   results: Dict[str, Any], 
                                   quality: Dict[str, Any]) -> str:
        """Create interactive dashboard."""
```

**Design Patterns**:
- **Builder Pattern**: Construct complex visualizations step by step
- **Factory Pattern**: Create different types of visualizations
- **Observer Pattern**: Update visualizations when data changes

#### **Utilities (`src/utils.py`)**

**Purpose**: Provide common utility functions used across the project.

**Responsibilities**:
- File and directory operations
- Image validation and manipulation
- Progress tracking and logging
- Error handling and recovery

**Key Functions**:
```python
def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if necessary."""
    
def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of file."""
    
def is_valid_image(file_path: str) -> bool:
    """Check if file is a valid image."""
    
def find_duplicates(file_paths: List[str]) -> List[List[str]]:
    """Find duplicate files based on MD5 hash."""
    
def create_progress_bar(total: int, description: str = "") -> tqdm:
    """Create progress bar for long-running operations."""
```

**Design Patterns**:
- **Utility Pattern**: Static functions for common operations
- **Singleton Pattern**: Shared resources like logging configuration

### **2. Configuration Management (`config/`)**

#### **Configuration Structure**

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

#### **Configuration Management**

```python
# config/dataset_config.py
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    
def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    
def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
```

**Design Patterns**:
- **Configuration Pattern**: Centralized configuration management
- **Validation Pattern**: Configuration validation and error checking
- **Default Pattern**: Sensible defaults with override capability

### **3. Scripts (`scripts/`)**

#### **Main Processing Scripts**

```python
# scripts/combine_datasets.py
def main():
    """Main entry point for dataset combination."""
    config = load_config(args.config)
    combiner = DatasetCombiner(config)
    results = combiner.combine_all_datasets()
    save_results(results, args.output)

# scripts/setup_datasets.py
def main():
    """Setup and validate source datasets."""
    config = load_config(args.config)
    setup_datasets(config, args.source, args.output)

# scripts/validate_combination.py
def main():
    """Validate combined dataset quality."""
    config = load_config(args.config)
    validation = validate_combined_dataset(config, args.dataset)
    print_validation_report(validation)
```

**Design Patterns**:
- **Command Pattern**: Scripts as command-line interfaces
- **Template Pattern**: Common script structure and error handling

### **4. Sample Project (`sample_segmentation_project/`)**

#### **Project Structure**

```
sample_segmentation_project/
├── config.py              # Configuration management
├── dataset.py             # Dataset loading and preprocessing
├── model.py               # U-Net model architecture
├── train_segmentation.py  # Training pipeline
├── evaluate_model.py      # Model evaluation
└── test_setup.py         # Setup validation
```

#### **Key Components**

```python
# sample_segmentation_project/config.py
class Config:
    """Configuration for segmentation project."""
    
    # Dataset paths
    DATASET_PATH = "/path/to/combined/dataset"
    
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

# sample_segmentation_project/dataset.py
class AgriculturalSegmentationDataset(Dataset):
    """Dataset for agricultural image segmentation."""
    
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, mask = self._load_sample(idx)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return {'image': image, 'mask': mask}

# sample_segmentation_project/model.py
class UNet(nn.Module):
    """U-Net architecture for image segmentation."""
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = self._create_encoder(in_channels)
        self.decoder = self._create_decoder(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        encoder_features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_features.append(x)
            x = F.max_pool2d(x, 2)
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, encoder_features[-(i+1)]], dim=1)
            x = decoder_block(x)
        
        return x

# sample_segmentation_project/train_segmentation.py
class Trainer:
    """Training pipeline for segmentation model."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = create_model()
        self.criterion = create_loss_function()
        self.optimizer = create_optimizer(self.model)
        self.scheduler = create_scheduler(self.optimizer)
        self.train_loader, self.val_loader = create_data_loaders()
        self.writer = SummaryWriter(config.LOG_DIR)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        
    def train(self) -> None:
        """Run complete training pipeline."""
```

**Design Patterns**:
- **MVC Pattern**: Model (U-Net), View (TensorBoard), Controller (Trainer)
- **Strategy Pattern**: Different loss functions and optimizers
- **Observer Pattern**: Training progress monitoring
- **Factory Pattern**: Model and component creation

## 🔄 **Data Flow**

### **1. Dataset Combination Flow**

```
Source Datasets → Validation → Preprocessing → Combination → Quality Control → Output
      │              │            │            │            │              │
      ▼              ▼            ▼            ▼            ▼              ▼
   PhenoBench    Format      Resize &      Merge &      Validation    Combined
   Capsicum      Check       Convert       Split        & Reports     Dataset
   Vineyard      & Pair      & Augment     & Balance    & Metrics     (62K+ images)
```

### **2. Processing Pipeline**

```
Input Image → Load → Validate → Resize → Convert → Augment → Save
     │         │       │        │        │         │        │
     ▼         ▼       ▼        ▼        ▼         ▼        ▼
  PNG/JPG   PIL/     Format   512x512   PNG      Rotate   Output
  Various   OpenCV   Check    Standard  Format   Flip     Directory
  Sizes     Load     & Pair   Size      Convert  Bright   Structure
```

### **3. Training Flow**

```
Combined Dataset → Data Loader → Model → Loss → Optimizer → Backprop → Checkpoint
       │              │          │      │      │          │          │
       ▼              ▼          ▼      ▼      ▼          ▼          ▼
   Train/Val/Test   Batches    U-Net   Dice   Adam       Gradients  Save Best
   Split (70/20/10) Augmented  (31M)   + BCE  Optimizer  Update     Model
```

### **4. Error Handling Flow**

```
Error Occurrence → Log Error → Attempt Recovery → Fallback Strategy → Continue/Fail
       │             │            │              │                  │
       ▼             ▼            ▼              ▼                  ▼
   File Not      Error Log    Retry with     Use Default        Skip Item
   Found         with Stack   Different      Configuration      or Stop
   Memory        Trace        Parameters     or Alternative     Processing
   Error         & Context    or Smaller     Implementation
```

## 🎨 **Design Patterns**

### **1. Architectural Patterns**

#### **Pipeline Pattern**
```python
class ProcessingPipeline:
    """Execute processing steps in sequence."""
    
    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps
    
    def execute(self, data: Any) -> Any:
        """Execute pipeline steps sequentially."""
        result = data
        for step in self.steps:
            try:
                result = step.process(result)
            except Exception as e:
                self._handle_error(step, e, result)
        return result
```

#### **Factory Pattern**
```python
class DatasetLoaderFactory:
    """Create appropriate dataset loader."""
    
    @staticmethod
    def create_loader(dataset_type: str, config: Dict[str, Any]) -> BaseDatasetLoader:
        if dataset_type == "phenobench":
            return PhenoBenchLoader(config)
        elif dataset_type == "capsicum":
            return CapsicumLoader(config)
        elif dataset_type == "vineyard":
            return VineyardLoader(config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
```

#### **Strategy Pattern**
```python
class ProcessingStrategy(ABC):
    """Abstract processing strategy."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data according to strategy."""
        pass

class FastProcessingStrategy(ProcessingStrategy):
    """Fast processing with minimal quality checks."""
    
    def process(self, data: Any) -> Any:
        # Fast processing implementation
        pass

class QualityProcessingStrategy(ProcessingStrategy):
    """Quality-focused processing with comprehensive checks."""
    
    def process(self, data: Any) -> Any:
        # Quality-focused processing implementation
        pass
```

### **2. Behavioral Patterns**

#### **Observer Pattern**
```python
class ProgressObserver(ABC):
    """Abstract progress observer."""
    
    @abstractmethod
    def update(self, progress: float, message: str):
        """Update progress information."""
        pass

class ConsoleProgressObserver(ProgressObserver):
    """Console-based progress display."""
    
    def update(self, progress: float, message: str):
        print(f"Progress: {progress:.1f}% - {message}")

class FileProgressObserver(ProgressObserver):
    """File-based progress logging."""
    
    def update(self, progress: float, message: str):
        with open("progress.log", "a") as f:
            f.write(f"{progress:.1f}% - {message}\n")
```

#### **Command Pattern**
```python
class ProcessingCommand(ABC):
    """Abstract processing command."""
    
    @abstractmethod
    def execute(self) -> Any:
        """Execute the command."""
        pass
    
    @abstractmethod
    def undo(self) -> Any:
        """Undo the command."""
        pass

class ResizeImageCommand(ProcessingCommand):
    """Command to resize an image."""
    
    def __init__(self, image: np.ndarray, target_size: Tuple[int, int]):
        self.image = image
        self.target_size = target_size
        self.original_size = image.shape[:2]
    
    def execute(self) -> np.ndarray:
        return cv2.resize(self.image, self.target_size)
    
    def undo(self) -> np.ndarray:
        return cv2.resize(self.image, self.original_size)
```

### **3. Structural Patterns**

#### **Adapter Pattern**
```python
class DatasetAdapter:
    """Adapt different dataset formats to common interface."""
    
    def __init__(self, dataset: Any):
        self.dataset = dataset
    
    def get_images(self) -> List[str]:
        """Get image paths in common format."""
        if hasattr(self.dataset, 'get_image_paths'):
            return self.dataset.get_image_paths()
        elif hasattr(self.dataset, 'images'):
            return self.dataset.images
        else:
            raise NotImplementedError("Dataset format not supported")
    
    def get_annotations(self) -> List[str]:
        """Get annotation paths in common format."""
        if hasattr(self.dataset, 'get_annotation_paths'):
            return self.dataset.get_annotation_paths()
        elif hasattr(self.dataset, 'annotations'):
            return self.dataset.annotations
        else:
            raise NotImplementedError("Dataset format not supported")
```

#### **Decorator Pattern**
```python
class ProcessingDecorator:
    """Add processing capabilities to base processor."""
    
    def __init__(self, processor: ImagePreprocessor):
        self.processor = processor
    
    def process_with_validation(self, image: np.ndarray) -> np.ndarray:
        """Process image with additional validation."""
        # Pre-validation
        if not self._validate_input(image):
            raise ValueError("Invalid input image")
        
        # Process
        result = self.processor.process_image(image)
        
        # Post-validation
        if not self._validate_output(result):
            raise ValueError("Invalid output image")
        
        return result
    
    def _validate_input(self, image: np.ndarray) -> bool:
        """Validate input image."""
        return image is not None and image.size > 0
    
    def _validate_output(self, image: np.ndarray) -> bool:
        """Validate output image."""
        return image is not None and image.size > 0
```

## ⚡ **Performance Considerations**

### **1. Memory Management**

#### **Batch Processing**
```python
def process_in_batches(files: List[str], batch_size: int = 1000):
    """Process files in batches to manage memory."""
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        process_batch(batch)
        gc.collect()  # Force garbage collection
```

#### **Memory-Efficient Loading**
```python
def load_image_efficiently(file_path: str) -> np.ndarray:
    """Load image with memory optimization."""
    # Use PIL for memory-efficient loading
    with Image.open(file_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
```

### **2. Parallel Processing**

#### **Multiprocessing**
```python
from multiprocessing import Pool, cpu_count

def process_parallel(files: List[str], num_processes: int = None):
    """Process files in parallel."""
    if num_processes is None:
        num_processes = cpu_count()
    
    with Pool(num_processes) as pool:
        results = pool.map(process_single_file, files)
    
    return results
```

#### **Async Processing**
```python
import asyncio
import aiofiles

async def process_file_async(file_path: str):
    """Process single file asynchronously."""
    async with aiofiles.open(file_path, 'rb') as f:
        content = await f.read()
        # Process content
        return processed_result

async def process_files_async(file_paths: List[str]):
    """Process multiple files asynchronously."""
    tasks = [process_file_async(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    return results
```

### **3. Caching and Optimization**

#### **Result Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def process_image_cached(image_hash: str, target_size: Tuple[int, int]):
    """Cache processed image results."""
    # Process image logic
    pass
```

#### **Lazy Loading**
```python
class LazyDatasetLoader:
    """Load dataset data only when needed."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._images = None
        self._annotations = None
    
    @property
    def images(self) -> List[str]:
        """Load images only when accessed."""
        if self._images is None:
            self._images = self._load_images()
        return self._images
    
    @property
    def annotations(self) -> List[str]:
        """Load annotations only when accessed."""
        if self._annotations is None:
            self._annotations = self._load_annotations()
        return self._annotations
```

## 📈 **Scalability**

### **1. Horizontal Scaling**

#### **Distributed Processing**
```python
class DistributedProcessor:
    """Distribute processing across multiple machines."""
    
    def __init__(self, worker_nodes: List[str]):
        self.worker_nodes = worker_nodes
        self.task_queue = Queue()
        self.result_queue = Queue()
    
    def distribute_tasks(self, tasks: List[Any]):
        """Distribute tasks across worker nodes."""
        # Split tasks among workers
        task_chunks = np.array_split(tasks, len(self.worker_nodes))
        
        # Send tasks to workers
        for i, worker in enumerate(self.worker_nodes):
            self._send_tasks_to_worker(worker, task_chunks[i])
    
    def collect_results(self) -> List[Any]:
        """Collect results from all workers."""
        results = []
        for _ in range(len(self.worker_nodes)):
            result = self.result_queue.get()
            results.extend(result)
        return results
```

#### **Load Balancing**
```python
class LoadBalancer:
    """Balance processing load across workers."""
    
    def __init__(self, workers: List[Worker]):
        self.workers = workers
        self.worker_loads = {worker.id: 0 for worker in workers}
    
    def get_least_loaded_worker(self) -> Worker:
        """Get worker with lowest current load."""
        return min(self.workers, key=lambda w: self.worker_loads[w.id])
    
    def assign_task(self, task: Any) -> Worker:
        """Assign task to least loaded worker."""
        worker = self.get_least_loaded_worker()
        self.worker_loads[worker.id] += 1
        worker.assign_task(task)
        return worker
```

### **2. Vertical Scaling**

#### **Resource Optimization**
```python
class ResourceManager:
    """Manage system resources efficiently."""
    
    def __init__(self):
        self.memory_limit = self._get_memory_limit()
        self.cpu_limit = self._get_cpu_limit()
    
    def optimize_batch_size(self, image_size: Tuple[int, int]) -> int:
        """Calculate optimal batch size based on available memory."""
        image_memory = image_size[0] * image_size[1] * 3 * 4  # RGB float32
        available_memory = self.memory_limit * 0.8  # Use 80% of available memory
        return int(available_memory / image_memory)
    
    def optimize_workers(self) -> int:
        """Calculate optimal number of worker processes."""
        return min(self.cpu_limit, 8)  # Cap at 8 workers
```

#### **Memory Pooling**
```python
class MemoryPool:
    """Reuse memory allocations to reduce overhead."""
    
    def __init__(self, pool_size: int = 100):
        self.pool_size = pool_size
        self.available_buffers = []
        self.used_buffers = set()
    
    def get_buffer(self, size: Tuple[int, int, int]) -> np.ndarray:
        """Get buffer from pool or create new one."""
        for buffer in self.available_buffers:
            if buffer.shape == size:
                self.available_buffers.remove(buffer)
                self.used_buffers.add(buffer)
                return buffer
        
        # Create new buffer if none available
        buffer = np.empty(size, dtype=np.uint8)
        self.used_buffers.add(buffer)
        return buffer
    
    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to pool."""
        if buffer in self.used_buffers:
            self.used_buffers.remove(buffer)
            if len(self.available_buffers) < self.pool_size:
                self.available_buffers.append(buffer)
```

## 🔒 **Security**

### **1. Input Validation**

#### **File Validation**
```python
def validate_file_path(file_path: str) -> bool:
    """Validate file path for security."""
    # Check for path traversal attacks
    if '..' in file_path or file_path.startswith('/'):
        return False
    
    # Check file extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
        return False
    
    return True

def validate_image_content(file_path: str) -> bool:
    """Validate image file content."""
    try:
        with Image.open(file_path) as img:
            # Check image dimensions
            if img.size[0] > 10000 or img.size[1] > 10000:
                return False
            
            # Check file size
            if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB limit
                return False
            
            return True
    except Exception:
        return False
```

#### **Configuration Validation**
```python
def validate_config_security(config: Dict[str, Any]) -> bool:
    """Validate configuration for security issues."""
    # Check for dangerous paths
    dangerous_paths = ['/', '/etc', '/var', '/usr', '/home']
    for path in [config.get('output_dir', ''), config.get('source_dir', '')]:
        for dangerous in dangerous_paths:
            if path.startswith(dangerous):
                return False
    
    # Check for dangerous commands
    dangerous_commands = ['rm', 'del', 'format', 'shutdown']
    for value in str(config).lower().split():
        for dangerous in dangerous_commands:
            if dangerous in value:
                return False
    
    return True
```

### **2. Access Control**

#### **File Permissions**
```python
def set_secure_permissions(file_path: str):
    """Set secure file permissions."""
    # Set restrictive permissions
    os.chmod(file_path, 0o600)  # Owner read/write only
    
    # Set ownership to current user
    os.chown(file_path, os.getuid(), os.getgid())

def create_secure_directory(dir_path: str):
    """Create directory with secure permissions."""
    os.makedirs(dir_path, mode=0o700, exist_ok=True)
    os.chown(dir_path, os.getuid(), os.getgid())
```

#### **Resource Limits**
```python
import resource

def set_resource_limits():
    """Set resource limits for security."""
    # Limit file size
    resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))  # 100MB
    
    # Limit number of open files
    resource.setrlimit(resource.RLIMIT_NOFILE, (1000, 1000))
    
    # Limit memory usage
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024))  # 2GB
```

## 🚀 **Deployment**

### **1. Containerization**

#### **Docker Configuration**
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["python", "scripts/combine_datasets.py"]
```

#### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  dataset-processor:
    build: .
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app
    command: python scripts/combine_datasets.py
    
  tensorboard:
    build: .
    volumes:
      - ./logs:/app/logs
    ports:
      - "6006:6006"
    command: tensorboard --logdir logs --host 0.0.0.0
```

### **2. Cloud Deployment**

#### **AWS Configuration**
```python
# aws_config.py
import boto3
from botocore.exceptions import ClientError

class AWSProcessor:
    """Process datasets using AWS services."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.s3 = boto3.client('s3', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)
        self.batch = boto3.client('batch', region_name=region)
    
    def upload_dataset(self, local_path: str, bucket: str, key: str):
        """Upload dataset to S3."""
        try:
            self.s3.upload_file(local_path, bucket, key)
            print(f"Dataset uploaded to s3://{bucket}/{key}")
        except ClientError as e:
            print(f"Upload failed: {e}")
    
    def process_on_ec2(self, dataset_s3_path: str, instance_type: str = 't3.large'):
        """Process dataset on EC2 instance."""
        # Launch EC2 instance
        # Download dataset from S3
        # Process dataset
        # Upload results to S3
        # Terminate instance
        pass
```

#### **Google Cloud Configuration**
```python
# gcp_config.py
from google.cloud import storage
from google.cloud import compute_v1

class GCPProcessor:
    """Process datasets using Google Cloud services."""
    
    def __init__(self, project_id: str):
        self.storage_client = storage.Client(project=project_id)
        self.compute_client = compute_v1.InstancesClient()
        self.project_id = project_id
    
    def upload_dataset(self, local_path: str, bucket_name: str, blob_name: str):
        """Upload dataset to Google Cloud Storage."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"Dataset uploaded to gs://{bucket_name}/{blob_name}")
    
    def process_on_compute(self, dataset_gcs_path: str, machine_type: str = 'n1-standard-2'):
        """Process dataset on Google Compute Engine."""
        # Create compute instance
        # Download dataset from GCS
        # Process dataset
        # Upload results to GCS
        # Delete instance
        pass
```

### **3. Monitoring and Logging**

#### **Structured Logging**
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured logging for better monitoring."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Add JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(name)s"}'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_processing_step(self, step: str, status: str, details: Dict[str, Any]):
        """Log processing step with structured information."""
        log_data = {
            'step': step,
            'status': status,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with context."""
        log_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.error(json.dumps(log_data))
```

#### **Performance Monitoring**
```python
import time
import psutil
import threading

class PerformanceMonitor:
    """Monitor system performance during processing."""
    
    def __init__(self):
        self.metrics = {}
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Store metrics
            self.metrics[timestamp] = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'disk_percent': disk.percent,
                'disk_used': disk.used
            }
            
            time.sleep(5)  # Update every 5 seconds
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {}
        
        timestamps = list(self.metrics.keys())
        cpu_values = [m['cpu_percent'] for m in self.metrics.values()]
        memory_values = [m['memory_percent'] for m in self.metrics.values()]
        
        return {
            'duration': max(timestamps) - min(timestamps),
            'avg_cpu': sum(cpu_values) / len(cpu_values),
            'max_cpu': max(cpu_values),
            'avg_memory': sum(memory_values) / len(memory_values),
            'max_memory': max(memory_values),
            'sample_count': len(self.metrics)
        }
```

---

## 🎯 **Architecture Summary**

### **Key Design Principles**

1. **Modularity**: Clear separation of concerns and responsibilities
2. **Extensibility**: Easy to add new dataset types and processing steps
3. **Reliability**: Robust error handling and recovery mechanisms
4. **Performance**: Efficient processing and memory management
5. **Security**: Input validation and access control
6. **Scalability**: Support for horizontal and vertical scaling

### **Technology Stack**

- **Language**: Python 3.8+
- **Core Libraries**: NumPy, OpenCV, PIL, PyTorch
- **Data Processing**: Pandas, SciPy, Albumentations
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Configuration**: PyYAML, ConfigArgParse
- **Testing**: Pytest, Coverage
- **Documentation**: Sphinx, Markdown
- **Deployment**: Docker, Cloud platforms

### **Architecture Benefits**

- **Maintainable**: Clear structure and separation of concerns
- **Testable**: Modular design enables comprehensive testing
- **Deployable**: Containerized and cloud-ready
- **Extensible**: Easy to add new features and datasets
- **Performant**: Optimized for speed and memory efficiency
- **Reliable**: Robust error handling and validation

---

**This architecture guide provides a comprehensive overview of the system design. For implementation details, refer to the source code and API documentation.** 🚀

---

<div align="center">

**Need More Details?** Check our [Complete Documentation](README.md) or [Source Code](https://github.com/selfishout/agricultural-dataset-combination)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/selfishout)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-torabi)

</div>
