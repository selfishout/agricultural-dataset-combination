# 🔧 Troubleshooting Guide

Comprehensive troubleshooting guide for the Agricultural Dataset Combination project. This guide covers common issues, error messages, and solutions.

## 📋 **Quick Diagnosis**

### **Common Symptoms**

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Import errors | Python path issues | Check `PYTHONPATH` and virtual environment |
| Memory errors | Insufficient RAM | Reduce batch size or use chunked processing |
| File not found | Path configuration | Verify paths in `config/dataset_config.yaml` |
| Permission denied | File permissions | Check file/directory permissions |
| Slow processing | Resource constraints | Optimize batch size and worker count |

### **Emergency Commands**

```bash
# Check system resources
free -h && df -h && nproc

# Check Python environment
python --version && which python && pip list

# Check project structure
ls -la && find . -name "*.py" | head -10

# Check logs
tail -f logs/*.log 2>/dev/null || echo "No logs found"
```

## 🐛 **Common Issues and Solutions**

### **1. Import and Module Errors**

#### **Issue: ModuleNotFoundError**

```python
# Error: ModuleNotFoundError: No module named 'src'
Traceback (most recent call last):
  File "scripts/combine_datasets.py", line 5, in <module>
    from src.dataset_combiner import DatasetCombiner
ModuleNotFoundError: No module named 'src'
```

**Solutions:**

```bash
# Solution 1: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solution 2: Install in development mode
pip install -e .

# Solution 3: Run from project root
cd /path/to/project/root
python scripts/combine_datasets.py

# Solution 4: Use absolute imports
python -c "import sys; sys.path.append('.'); from src.dataset_combiner import DatasetCombiner"
```

#### **Issue: Relative Import Errors**

```python
# Error: ImportError: attempted relative import with no known parent package
from .utils import setup_logging
ImportError: attempted relative import with no known parent package
```

**Solutions:**

```python
# Fix 1: Change relative imports to absolute
# Before: from .utils import setup_logging
# After: from utils import setup_logging

# Fix 2: Add __init__.py files
touch src/__init__.py
touch tests/__init__.py

# Fix 3: Use absolute imports with project root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logging
```

### **2. File and Path Issues**

#### **Issue: FileNotFoundError**

```python
# Error: FileNotFoundError: [Errno 2] No such file or directory
FileNotFoundError: [Errno 2] No such file or directory: 'config/dataset_config.yaml'
```

**Solutions:**

```bash
# Solution 1: Check file existence
ls -la config/
find . -name "dataset_config.yaml"

# Solution 2: Create missing directories
mkdir -p config data/{raw,processed,intermediate,final} logs

# Solution 3: Check working directory
pwd && ls -la

# Solution 4: Use absolute paths
python scripts/combine_datasets.py --config /absolute/path/to/config.yaml
```

#### **Issue: Permission Denied**

```python
# Error: PermissionError: [Errno 13] Permission denied
PermissionError: [Errno 13] Permission denied: '/data/output'
```

**Solutions:**

```bash
# Solution 1: Check permissions
ls -la /data/
whoami && groups

# Solution 2: Fix permissions
sudo chown -R $USER:$USER /data/
chmod -R 755 /data/

# Solution 3: Use user-writable directory
mkdir -p ~/agricultural_data
# Update config to use ~/agricultural_data

# Solution 4: Run with appropriate user
sudo -u appropriate_user python scripts/combine_datasets.py
```

### **3. Memory and Resource Issues**

#### **Issue: MemoryError**

```python
# Error: MemoryError: Unable to allocate array
MemoryError: Unable to allocate array with shape (10000, 1024, 1024, 3)
```

**Solutions:**

```python
# Solution 1: Reduce batch size
config['processing']['batch_size'] = 4  # Instead of 16

# Solution 2: Use chunked processing
def process_in_chunks(items, chunk_size=1000):
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        process_chunk(chunk)
        gc.collect()  # Force garbage collection

# Solution 3: Optimize image size
config['processing']['target_size'] = [512, 512]  # Instead of [1024, 1024]

# Solution 4: Use memory-efficient data types
import numpy as np
# Use float32 instead of float64
images = np.array(images, dtype=np.float32)
```

#### **Issue: Out of Memory (OOM)**

```python
# Error: RuntimeError: CUDA out of memory
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

```python
# Solution 1: Clear GPU memory
import torch
torch.cuda.empty_cache()

# Solution 2: Reduce batch size for GPU
config['processing']['batch_size'] = 2  # Very small for GPU

# Solution 3: Use CPU instead of GPU
device = torch.device('cpu')
# Or set environment variable
# export CUDA_VISIBLE_DEVICES=""

# Solution 4: Gradient checkpointing (for training)
model.use_checkpoint = True
```

### **4. Configuration Issues**

#### **Issue: Configuration Loading Errors**

```python
# Error: yaml.constructor.ConstructorError: could not determine a constructor
yaml.constructor.ConstructorError: could not determine a constructor for the tag '!include'
```

**Solutions:**

```yaml
# Fix 1: Remove unsupported YAML features
# Before: !include other_file.yaml
# After: # Include other_file.yaml manually

# Fix 2: Use simple YAML syntax
storage:
  output_dir: "/data/output"
  intermediate_dir: "/data/intermediate"

processing:
  target_size: [512, 512]
  image_format: "PNG"
```

```python
# Fix 3: Load config manually
import yaml
with open('config/dataset_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Fix 4: Use environment variables
import os
config = {
    'storage': {
        'output_dir': os.environ.get('OUTPUT_DIR', '/data/output')
    }
}
```

#### **Issue: Missing Configuration Keys**

```python
# Error: KeyError: 'processing_info'
KeyError: 'processing_info'
```

**Solutions:**

```python
# Fix 1: Add safe dictionary access
results = combiner.combine_all_datasets()
processing_info = results.get('processing_info', {})
quality_control = processing_info.get('quality_control', {})

# Fix 2: Check key existence
if 'processing_info' in results:
    # Process processing_info
    pass
else:
    # Handle missing key
    print("Warning: processing_info not found in results")

# Fix 3: Provide default values
def get_config_value(config, key, default=None):
    """Safely get configuration value with default."""
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default
```

### **5. Dataset Processing Issues**

#### **Issue: Corrupted Image Files**

```python
# Error: OSError: cannot identify image file
OSError: cannot identify image file '/path/to/image.jpg'
```

**Solutions:**

```python
# Solution 1: Validate image files
from PIL import Image
import os

def validate_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

# Solution 2: Skip corrupted files
valid_images = []
for img_path in image_paths:
    if validate_image(img_path):
        valid_images.append(img_path)
    else:
        print(f"Skipping corrupted image: {img_path}")

# Solution 3: Check file headers
def check_file_header(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(8)
        if header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            return True
        elif header.startswith(b'\xff\xd8\xff'):  # JPEG
            return True
        return False
```

#### **Issue: Annotation Mismatch**

```python
# Error: ValueError: Found input variables with inconsistent numbers of samples
ValueError: Found input variables with inconsistent numbers of samples
```

**Solutions:**

```python
# Solution 1: Verify image-annotation pairs
def verify_pairs(image_dir, annotation_dir):
    images = set(os.listdir(image_dir))
    annotations = set(os.listdir(annotation_dir))
    
    # Find missing annotations
    missing_annotations = images - annotations
    if missing_annotations:
        print(f"Missing annotations for: {missing_annotations}")
    
    # Find orphaned annotations
    orphaned_annotations = annotations - images
    if orphaned_annotations:
        print(f"Orphaned annotations: {orphaned_annotations}")
    
    return len(missing_annotations) == 0 and len(orphaned_annotations) == 0

# Solution 2: Create mapping file
def create_image_annotation_map(image_dir, annotation_dir):
    mapping = {}
    for img_file in os.listdir(image_dir):
        base_name = os.path.splitext(img_file)[0]
        annotation_file = f"{base_name}.png"  # Adjust extension as needed
        
        if os.path.exists(os.path.join(annotation_dir, annotation_file)):
            mapping[img_file] = annotation_file
    
    return mapping
```

### **6. Performance Issues**

#### **Issue: Slow Processing**

```python
# Problem: Processing is very slow
# Expected: 1000 images/hour
# Actual: 100 images/hour
```

**Solutions:**

```python
# Solution 1: Optimize batch size
def find_optimal_batch_size():
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    # Assume 2GB per batch
    return max(1, int(memory_gb / 2))

# Solution 2: Use parallel processing
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_parallel(items, func, max_workers=None):
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    
    return results

# Solution 3: Optimize image loading
def load_image_optimized(image_path):
    # Use PIL for faster loading
    from PIL import Image
    import numpy as np
    
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize during loading
        img = img.resize((512, 512), Image.LANCZOS)
        return np.array(img)
```

#### **Issue: High Memory Usage**

```python
# Problem: Memory usage keeps growing
# Expected: Stable memory usage
# Actual: Memory usage increases over time
```

**Solutions:**

```python
# Solution 1: Force garbage collection
import gc

def process_with_cleanup(items):
    for i, item in enumerate(items):
        process_item(item)
        
        # Clean up every 100 items
        if i % 100 == 0:
            gc.collect()
            print(f"Processed {i} items, memory cleaned")

# Solution 2: Use context managers
def process_images(image_paths):
    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Process image
            processed = process_single_image(img)
            yield processed  # Generator to avoid storing all in memory

# Solution 3: Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
```

## 🔍 **Debugging Techniques**

### **1. Logging and Debugging**

#### **Enhanced Logging Setup**

```python
# src/debug_logging.py
import logging
import traceback
import sys
from datetime import datetime

def setup_debug_logging():
    """Setup comprehensive debugging logging."""
    
    # Create logger
    logger = logging.getLogger('agricultural_dataset')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler('debug.log')
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def log_exception(logger, e, context=""):
    """Log exception with full traceback and context."""
    logger.error(f"Exception in {context}: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Log system information
    import platform
    logger.error(f"System: {platform.system()} {platform.release()}")
    logger.error(f"Python: {sys.version}")
```

#### **Debug Decorators**

```python
# src/debug_decorators.py
import functools
import time
import logging
import traceback

def debug_function(func):
    """Decorator to add debugging information to functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('agricultural_dataset')
        
        # Log function call
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Log successful execution
            logger.debug(f"{func.__name__} completed in {end_time - start_time:.2f}s")
            return result
            
        except Exception as e:
            # Log exception
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    return wrapper

def memory_tracker(func):
    """Decorator to track memory usage of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_diff = memory_after - memory_before
        
        print(f"{func.__name__}: Memory change: {memory_diff:+.2f} MB")
        return result
    
    return wrapper
```

### **2. Interactive Debugging**

#### **IPython Debugger Integration**

```python
# src/interactive_debug.py
import ipdb
import sys

def interactive_debug(condition=True):
    """Drop into IPython debugger if condition is True."""
    if condition:
        ipdb.set_trace()

def debug_on_error():
    """Automatically drop into debugger on any error."""
    def excepthook(type, value, traceback):
        print(f"\nError: {type.__name__}: {value}")
        ipdb.post_mortem(traceback)
    
    sys.excepthook = excepthook

# Usage in your code
def process_dataset():
    try:
        # Your processing code
        result = some_function()
        interactive_debug(result is None)  # Debug if result is None
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        interactive_debug()  # Always debug on error
        raise
```

#### **Debug Console**

```python
# src/debug_console.py
import cmd
import os
import sys
from pathlib import Path

class DatasetDebugConsole(cmd.Cmd):
    """Interactive debug console for dataset processing."""
    
    intro = 'Welcome to the Agricultural Dataset Debug Console. Type help or ? to list commands.\n'
    prompt = '(dataset-debug) '
    
    def __init__(self):
        super().__init__()
        self.current_dir = Path.cwd()
        self.config = {}
    
    def do_pwd(self, arg):
        """Show current working directory."""
        print(f"Current directory: {self.current_dir}")
    
    def do_ls(self, arg):
        """List files in current directory."""
        try:
            files = list(self.current_dir.iterdir())
            for file in files:
                print(f"{'d' if file.is_dir() else '-'} {file.name}")
        except Exception as e:
            print(f"Error listing directory: {e}")
    
    def do_cd(self, arg):
        """Change directory."""
        if not arg:
            print("Usage: cd <directory>")
            return
        
        new_dir = self.current_dir / arg
        if new_dir.exists() and new_dir.is_dir():
            self.current_dir = new_dir
            print(f"Changed to: {self.current_dir}")
        else:
            print(f"Directory not found: {new_dir}")
    
    def do_check_config(self, arg):
        """Check configuration file."""
        config_file = self.current_dir / 'config' / 'dataset_config.yaml'
        if config_file.exists():
            print(f"Config file exists: {config_file}")
            # You could add YAML parsing here
        else:
            print(f"Config file not found: {config_file}")
    
    def do_exit(self, arg):
        """Exit the debug console."""
        print("Goodbye!")
        return True

# Usage
if __name__ == '__main__':
    console = DatasetDebugConsole()
    console.cmdloop()
```

## 🧪 **Testing and Validation**

### **1. Unit Testing for Debugging**

```python
# tests/test_debug.py
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

class TestDatasetDebugging(unittest.TestCase):
    """Test debugging utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'storage': {
                'output_dir': self.temp_dir,
                'intermediate_dir': os.path.join(self.temp_dir, 'intermediate')
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_validation(self):
        """Test configuration validation."""
        from src.utils import validate_config
        
        # Test valid config
        self.assertTrue(validate_config(self.config))
        
        # Test invalid config
        invalid_config = {}
        self.assertFalse(validate_config(invalid_config))
    
    def test_file_validation(self):
        """Test file validation."""
        from src.utils import validate_file_path
        
        # Test valid file path
        valid_path = os.path.join(self.temp_dir, 'test.txt')
        with open(valid_path, 'w') as f:
            f.write('test')
        
        self.assertTrue(validate_file_path(valid_path))
        
        # Test invalid file path
        invalid_path = os.path.join(self.temp_dir, 'nonexistent.txt')
        self.assertFalse(validate_file_path(invalid_path))
    
    @patch('psutil.virtual_memory')
    def test_memory_monitoring(self, mock_memory):
        """Test memory monitoring."""
        from src.memory_manager import MemoryManager
        
        # Mock memory info
        mock_memory.return_value.percent = 85.0
        
        manager = MemoryManager(max_memory_percent=80.0)
        memory_info = manager.get_memory_info()
        
        self.assertEqual(memory_info['percent'], 85.0)
        self.assertTrue(manager._should_cleanup())

if __name__ == '__main__':
    unittest.main()
```

### **2. Integration Testing**

```python
# tests/test_integration.py
import unittest
import tempfile
import os
import shutil
from pathlib import Path

class TestDatasetIntegration(unittest.TestCase):
    """Test complete dataset processing pipeline."""
    
    def setUp(self):
        """Set up test environment with sample data."""
        self.test_dir = tempfile.mkdtemp()
        self.setup_test_data()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def setup_test_data(self):
        """Create sample dataset for testing."""
        # Create sample images
        images_dir = os.path.join(self.test_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Create sample annotations
        annotations_dir = os.path.join(self.test_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Create sample config
        config_dir = os.path.join(self.test_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        config_content = f"""
storage:
  output_dir: "{os.path.join(self.test_dir, 'output')}"
  intermediate_dir: "{os.path.join(self.test_dir, 'intermediate')}"
  backup_original: false

processing:
  target_size: [64, 64]
  image_format: "PNG"
  augmentation: false
  quality_check: true

splits:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
"""
        
        with open(os.path.join(config_dir, 'test_config.yaml'), 'w') as f:
            f.write(config_content)
    
    def test_complete_pipeline(self):
        """Test complete dataset processing pipeline."""
        from src.dataset_combiner import DatasetCombiner
        from config.dataset_config import load_config
        
        # Load test configuration
        config_path = os.path.join(self.test_dir, 'config', 'test_config.yaml')
        config = load_config(config_path)
        
        # Create combiner
        combiner = DatasetCombiner(config)
        
        # Test pipeline
        try:
            results = combiner.combine_all_datasets()
            self.assertIsNotNone(results)
            print(f"Pipeline completed successfully: {results}")
        except Exception as e:
            self.fail(f"Pipeline failed with error: {e}")
    
    def test_error_handling(self):
        """Test error handling in pipeline."""
        from src.dataset_combiner import DatasetCombiner
        
        # Test with invalid configuration
        invalid_config = {'invalid': 'config'}
        
        with self.assertRaises(Exception):
            combiner = DatasetCombiner(invalid_config)
            combiner.combine_all_datasets()

if __name__ == '__main__':
    unittest.main()
```

## 📊 **Performance Profiling**

### **1. Memory Profiling**

```python
# src/memory_profiler.py
import psutil
import os
import time
from typing import Dict, Any
import gc

class MemoryProfiler:
    """Memory usage profiling and analysis."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
    
    def take_snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        memory_info = self.process.memory_info()
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def print_snapshot(self, snapshot: Dict[str, Any]):
        """Print memory snapshot information."""
        print(f"Memory Snapshot - {snapshot['label']}")
        print(f"  RSS: {snapshot['rss_mb']:.2f} MB")
        print(f"  VMS: {snapshot['vms_mb']:.2f} MB")
        print(f"  Percent: {snapshot['percent']:.2f}%")
    
    def analyze_memory_growth(self):
        """Analyze memory growth patterns."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots for analysis")
            return
        
        print("\nMemory Growth Analysis:")
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            
            rss_diff = curr['rss_mb'] - prev['rss_mb']
            vms_diff = curr['vms_mb'] - prev['vms_mb']
            
            print(f"\n{prev['label']} -> {curr['label']}:")
            print(f"  RSS change: {rss_diff:+.2f} MB")
            print(f"  VMS change: {vms_diff:+.2f} MB")
    
    def force_cleanup(self):
        """Force memory cleanup."""
        print("Forcing memory cleanup...")
        gc.collect()
        
        # Take snapshot after cleanup
        cleanup_snapshot = self.take_snapshot("After cleanup")
        self.print_snapshot(cleanup_snapshot)
        
        return cleanup_snapshot

# Usage example
def profile_function(func, *args, **kwargs):
    """Profile memory usage of a function."""
    profiler = MemoryProfiler()
    
    # Take initial snapshot
    initial = profiler.take_snapshot("Before function call")
    profiler.print_snapshot(initial)
    
    # Call function
    result = func(*args, **kwargs)
    
    # Take final snapshot
    final = profiler.take_snapshot("After function call")
    profiler.print_snapshot(final)
    
    # Analyze growth
    profiler.analyze_memory_growth()
    
    return result
```

### **2. Performance Monitoring**

```python
# src/performance_monitor.py
import time
import functools
from typing import Callable, Any, Dict
import statistics

class PerformanceMonitor:
    """Performance monitoring and timing analysis."""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, func_name: str = None):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            name = func_name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                # Store timing data
                if name not in self.timings:
                    self.timings[name] = []
                    self.call_counts[name] = 0
                
                self.timings[name].append(execution_time)
                self.call_counts[name] += 1
                
                return result
            
            return wrapper
        return decorator
    
    def get_statistics(self, func_name: str) -> Dict[str, float]:
        """Get performance statistics for a function."""
        if func_name not in self.timings:
            return {}
        
        times = self.timings[func_name]
        return {
            'count': len(times),
            'total_time': sum(times),
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'median_time': statistics.median(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def print_summary(self):
        """Print performance summary for all monitored functions."""
        print("\nPerformance Summary:")
        print("=" * 50)
        
        for func_name in self.timings:
            stats = self.get_statistics(func_name)
            print(f"\n{func_name}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Total time: {stats['total_time']:.4f}s")
            print(f"  Average time: {stats['avg_time']:.4f}s")
            print(f"  Min time: {stats['min_time']:.4f}s")
            print(f"  Max time: {stats['max_time']:.4f}s")
            print(f"  Median time: {stats['median_time']:.4f}s")
            print(f"  Std dev: {stats['std_dev']:.4f}s")

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Usage example
@performance_monitor.time_function()
def slow_function():
    """Example function to monitor."""
    time.sleep(0.1)  # Simulate work
    return "done"

# Monitor multiple calls
for i in range(5):
    slow_function()

# Print summary
performance_monitor.print_summary()
```

## 🚨 **Emergency Recovery**

### **1. Data Recovery**

```python
# src/emergency_recovery.py
import os
import shutil
import json
from datetime import datetime
from pathlib import Path

class EmergencyRecovery:
    """Emergency data recovery utilities."""
    
    def __init__(self, backup_dir: str = "emergency_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, source_path: str, label: str = ""):
        """Create emergency backup of data."""
        source = Path(source_path)
        if not source.exists():
            print(f"Source path does not exist: {source_path}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{label}_{timestamp}" if label else timestamp
        backup_path = self.backup_dir / backup_name
        
        try:
            if source.is_file():
                shutil.copy2(source, backup_path)
            else:
                shutil.copytree(source, backup_path)
            
            print(f"Emergency backup created: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Failed to create backup: {e}")
            return None
    
    def list_backups(self):
        """List available emergency backups."""
        backups = []
        for item in self.backup_dir.iterdir():
            if item.is_dir() or item.is_file():
                stat = item.stat()
                backups.append({
                    'name': item.name,
                    'type': 'directory' if item.is_dir() else 'file',
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
        
        return sorted(backups, key=lambda x: x['modified'], reverse=True)
    
    def restore_backup(self, backup_name: str, target_path: str):
        """Restore data from emergency backup."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            print(f"Backup not found: {backup_name}")
            return False
        
        target = Path(target_path)
        
        try:
            if backup_path.is_file():
                shutil.copy2(backup_path, target)
            else:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(backup_path, target)
            
            print(f"Backup restored to: {target_path}")
            return True
        except Exception as e:
            print(f"Failed to restore backup: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 5):
        """Clean up old emergency backups."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            print("No old backups to clean up")
            return
        
        # Remove oldest backups
        for backup in backups[keep_count:]:
            backup_path = self.backup_dir / backup['name']
            try:
                if backup_path.is_file():
                    backup_path.unlink()
                else:
                    shutil.rmtree(backup_path)
                print(f"Removed old backup: {backup['name']}")
            except Exception as e:
                print(f"Failed to remove backup {backup['name']}: {e}")

# Usage example
def emergency_backup_data():
    """Create emergency backup of critical data."""
    recovery = EmergencyRecovery()
    
    # Backup critical directories
    critical_paths = [
        'data/processed',
        'config',
        'logs'
    ]
    
    for path in critical_paths:
        if os.path.exists(path):
            recovery.create_backup(path, f"critical_{os.path.basename(path)}")
    
    # List available backups
    backups = recovery.list_backups()
    print(f"Available backups: {len(backups)}")
    
    # Clean up old backups
    recovery.cleanup_old_backups(keep_count=3)
```

### **2. Process Recovery**

```python
# src/process_recovery.py
import psutil
import os
import signal
import time
from typing import List, Dict

class ProcessRecovery:
    """Process monitoring and recovery utilities."""
    
    def __init__(self, process_name: str = "python"):
        self.process_name = process_name
        self.target_scripts = [
            "combine_datasets.py",
            "setup_datasets.py",
            "validate_combination.py"
        ]
    
    def find_project_processes(self) -> List[Dict]:
        """Find running project processes."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
            try:
                if proc.info['name'] == self.process_name:
                    cmdline = proc.info['cmdline']
                    if cmdline and any(script in ' '.join(cmdline) for script in self.target_scripts):
                        processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline,
                            'status': proc.info['status'],
                            'memory_mb': proc.memory_info().rss / (1024 * 1024),
                            'cpu_percent': proc.cpu_percent()
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def kill_process(self, pid: int, force: bool = False):
        """Kill a process by PID."""
        try:
            proc = psutil.Process(pid)
            
            if force:
                proc.kill()
                print(f"Force killed process {pid}")
            else:
                proc.terminate()
                print(f"Terminated process {pid}")
                
                # Wait for graceful termination
                try:
                    proc.wait(timeout=10)
                except psutil.TimeoutExpired:
                    proc.kill()
                    print(f"Force killed process {pid} after timeout")
                    
        except psutil.NoSuchProcess:
            print(f"Process {pid} not found")
        except psutil.AccessDenied:
            print(f"Access denied to process {pid}")
    
    def restart_process(self, script_name: str, args: List[str] = None):
        """Restart a specific script."""
        # Kill existing processes
        processes = self.find_project_processes()
        for proc in processes:
            if script_name in ' '.join(proc['cmdline']):
                print(f"Killing existing {script_name} process: {proc['pid']}")
                self.kill_process(proc['pid'])
        
        # Wait for processes to terminate
        time.sleep(2)
        
        # Start new process
        cmd = ['python', f'scripts/{script_name}']
        if args:
            cmd.extend(args)
        
        try:
            import subprocess
            subprocess.Popen(cmd, cwd=os.getcwd())
            print(f"Started new {script_name} process")
        except Exception as e:
            print(f"Failed to start {script_name}: {e}")
    
    def monitor_and_recover(self, script_name: str, max_restarts: int = 3):
        """Monitor a script and restart if it fails."""
        restarts = 0
        
        while restarts < max_restarts:
            processes = self.find_project_processes()
            running = any(script_name in ' '.join(p['cmdline']) for p in processes)
            
            if not running:
                print(f"{script_name} not running, restarting...")
                self.restart_process(script_name)
                restarts += 1
                time.sleep(10)  # Wait before checking again
            else:
                print(f"{script_name} running normally")
                time.sleep(30)  # Check every 30 seconds
        
        print(f"Maximum restarts ({max_restarts}) reached for {script_name}")

# Usage example
def emergency_process_recovery():
    """Emergency process recovery for critical scripts."""
    recovery = ProcessRecovery()
    
    # Find all project processes
    processes = recovery.find_project_processes()
    print(f"Found {len(processes)} project processes:")
    
    for proc in processes:
        print(f"  PID {proc['pid']}: {' '.join(proc['cmdline'])}")
        print(f"    Status: {proc['status']}, Memory: {proc['memory_mb']:.1f} MB")
    
    # Check for stuck processes
    for proc in processes:
        if proc['status'] == 'zombie':
            print(f"Found zombie process {proc['pid']}, killing...")
            recovery.kill_process(proc['pid'], force=True)
    
    # Restart critical script if needed
    recovery.restart_process("combine_datasets.py")
```

---

## 🎯 **Troubleshooting Summary**

### **What We've Covered**

✅ **Common Issues**: Import, file, memory, configuration problems  
✅ **Debugging Techniques**: Logging, interactive debugging, profiling  
✅ **Testing**: Unit tests, integration tests for debugging  
✅ **Performance**: Memory and performance monitoring  
✅ **Emergency Recovery**: Data and process recovery utilities  

### **Quick Reference**

| Issue Type | First Action | Tool/Command |
|------------|--------------|--------------|
| Import errors | Check `PYTHONPATH` | `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` |
| File not found | Verify paths | `ls -la config/ && pwd` |
| Memory issues | Reduce batch size | Update config batch_size |
| Slow processing | Profile performance | Use `@performance_monitor.time_function()` |
| Process stuck | Check process status | `ps aux | grep python` |

### **Next Steps**

1. **Identify the issue** using the quick diagnosis section
2. **Apply the solution** from the common issues section
3. **Use debugging tools** if the issue persists
4. **Profile performance** for optimization opportunities
5. **Create tests** to prevent future occurrences

---

**This troubleshooting guide provides comprehensive solutions for common issues. If your problem isn't covered here, check the [GitHub Issues](https://github.com/selfishout/agricultural-dataset-combination/issues) or create a new one.** 🔧

---

<div align="center">

**Need More Help?** Check our [Complete Documentation](README.md) or [Source Code](https://github.com/selfishout/agricultural-dataset-combination)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/selfishout)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-torabi)

</div>
