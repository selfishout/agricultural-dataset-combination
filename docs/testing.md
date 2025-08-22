# 🧪 Testing Guide

Comprehensive testing guide for the Agricultural Dataset Combination project. This guide covers testing strategies, frameworks, and best practices.

## 📋 **Testing Overview**

### **Testing Strategy**

Our testing approach follows the **Testing Pyramid**:
- **Unit Tests** (70%): Fast, isolated tests for individual components
- **Integration Tests** (20%): Tests for component interactions
- **End-to-End Tests** (10%): Full workflow validation

### **Testing Goals**

1. **Reliability**: Ensure consistent behavior across environments
2. **Maintainability**: Catch regressions early
3. **Documentation**: Tests serve as living documentation
4. **Confidence**: Enable safe refactoring and updates

## 🏗️ **Testing Framework**

### **Core Testing Tools**

```bash
# Install testing dependencies
pip install -r requirements-dev.txt

# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_dataset_combiner.py -v
```

### **Test Structure**

```
tests/
├── __init__.py
├── test_dataset_combiner.py      # Core functionality tests
├── test_preprocessing.py         # Preprocessing tests
├── test_visualization.py         # Visualization tests
├── test_utils.py                 # Utility function tests
├── conftest.py                   # Shared fixtures
└── integration/                  # Integration tests
    ├── test_full_pipeline.py
    └── test_sample_project.py
```

## 🧩 **Unit Testing**

### **Dataset Combiner Tests**

```python
# tests/test_dataset_combiner.py
import pytest
from unittest.mock import Mock, patch
from src.dataset_combiner import DatasetCombiner

class TestDatasetCombiner:
    """Test DatasetCombiner functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'storage': {
                'output_dir': '/tmp/test_output',
                'intermediate_dir': '/tmp/test_intermediate'
            },
            'processing': {
                'target_size': [512, 512],
                'image_format': 'PNG'
            }
        }
    
    @pytest.fixture
    def combiner(self, mock_config):
        """Create DatasetCombiner instance for testing."""
        return DatasetCombiner(mock_config)
    
    def test_initialization(self, combiner, mock_config):
        """Test DatasetCombiner initialization."""
        assert combiner.config == mock_config
        assert combiner.output_dir == mock_config['storage']['output_dir']
    
    def test_combine_all_datasets(self, combiner):
        """Test complete dataset combination workflow."""
        with patch('src.dataset_combiner.PhenoBenchLoader') as mock_loader:
            mock_loader.return_value.load_images.return_value = ['/path/to/image1.png']
            mock_loader.return_value.load_annotations.return_value = ['/path/to/ann1.png']
            
            results = combiner.combine_all_datasets()
            
            assert 'statistics' in results
            assert 'processing_results' in results
            assert results['statistics']['total_images'] > 0
    
    def test_validation(self, combiner):
        """Test dataset validation."""
        validation = combiner.validate_combination()
        assert 'quality_metrics' in validation
        assert 'error_count' in validation
```

### **Preprocessing Tests**

```python
# tests/test_preprocessing.py
import pytest
import numpy as np
from src.preprocessing import ImagePreprocessor, AnnotationProcessor

class TestImagePreprocessor:
    """Test ImagePreprocessor functionality."""
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image for testing."""
        return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    @pytest.fixture
    def preprocessor(self):
        """Create ImagePreprocessor instance."""
        return ImagePreprocessor(target_size=(512, 512), format='PNG')
    
    def test_resize_image(self, preprocessor, sample_image):
        """Test image resizing."""
        resized = preprocessor.resize_image(sample_image)
        assert resized.shape == (512, 512, 3)
        assert resized.dtype == np.uint8
    
    def test_format_conversion(self, preprocessor, sample_image):
        """Test image format conversion."""
        converted = preprocessor.convert_format(sample_image)
        assert converted.shape == sample_image.shape
        assert converted.dtype == np.uint8
    
    def test_augmentation(self, preprocessor, sample_image):
        """Test data augmentation."""
        augmented, _ = preprocessor.apply_augmentation(sample_image)
        assert augmented.shape == sample_image.shape
        assert augmented.dtype == np.uint8

class TestAnnotationProcessor:
    """Test AnnotationProcessor functionality."""
    
    @pytest.fixture
    def sample_annotation(self):
        """Create sample annotation for testing."""
        return np.random.randint(0, 2, (1024, 1024), dtype=np.uint8)
    
    @pytest.fixture
    def processor(self):
        """Create AnnotationProcessor instance."""
        return AnnotationProcessor(target_size=(512, 512), format='PNG')
    
    def test_resize_annotation(self, processor, sample_annotation):
        """Test annotation resizing."""
        resized = processor.resize_annotation(sample_annotation)
        assert resized.shape == (512, 512)
        assert resized.dtype == np.uint8
    
    def test_annotation_validation(self, processor, sample_annotation):
        """Test annotation validation."""
        is_valid = processor.validate_annotation(sample_annotation)
        assert isinstance(is_valid, bool)
```

### **Utility Function Tests**

```python
# tests/test_utils.py
import pytest
import tempfile
import os
from src.utils import ensure_directory, calculate_md5, is_valid_image

class TestUtils:
    """Test utility functions."""
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, 'test_subdir')
            ensure_directory(test_dir)
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)
    
    def test_calculate_md5(self):
        """Test MD5 hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            f.flush()
            
            md5_hash = calculate_md5(f.name)
            assert len(md5_hash) == 32
            assert md5_hash.isalnum()
            
            os.unlink(f.name)
    
    def test_is_valid_image(self):
        """Test image validation."""
        # Test with non-existent file
        assert not is_valid_image('/nonexistent/file.png')
        
        # Test with valid image (create test image)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Create minimal PNG file
            f.write(b'\x89PNG\r\n\x1a\n')
            f.flush()
            
            # Note: This is a minimal test - real images would be better
            # In practice, you'd want to test with actual image files
            os.unlink(f.name)
```

## 🔗 **Integration Testing**

### **Full Pipeline Tests**

```python
# tests/integration/test_full_pipeline.py
import pytest
import tempfile
import shutil
import os
from src.dataset_combiner import DatasetCombiner
from config.dataset_config import load_config

class TestFullPipeline:
    """Test complete dataset combination pipeline."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_workspace):
        """Create test configuration."""
        return {
            'storage': {
                'output_dir': os.path.join(temp_workspace, 'output'),
                'intermediate_dir': os.path.join(temp_workspace, 'intermediate'),
                'backup_original': False
            },
            'processing': {
                'target_size': [256, 256],  # Smaller for faster testing
                'image_format': 'PNG',
                'augmentation': False,  # Disable for testing
                'quality_check': True,
                'batch_size': 2
            },
            'splits': {
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1
            }
        }
    
    def test_end_to_end_pipeline(self, test_config, temp_workspace):
        """Test complete end-to-end pipeline."""
        # Create test dataset structure
        test_dataset_dir = os.path.join(temp_workspace, 'test_dataset')
        os.makedirs(os.path.join(test_dataset_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(test_dataset_dir, 'annotations'), exist_ok=True)
        
        # Create test images and annotations (simplified)
        # In practice, you'd copy real test data here
        
        # Run pipeline
        combiner = DatasetCombiner(test_config)
        results = combiner.combine_all_datasets()
        
        # Validate results
        assert 'statistics' in results
        assert 'processing_results' in results
        assert os.path.exists(test_config['storage']['output_dir'])
        
        # Check output structure
        output_dir = test_config['storage']['output_dir']
        assert os.path.exists(os.path.join(output_dir, 'train'))
        assert os.path.exists(os.path.join(output_dir, 'val'))
        assert os.path.exists(os.path.join(output_dir, 'test'))
```

### **Sample Project Tests**

```python
# tests/integration/test_sample_project.py
import pytest
import tempfile
import shutil
import os
import sys

# Add sample project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sample_segmentation_project'))

class TestSampleProject:
    """Test sample segmentation project functionality."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_creation(self):
        """Test U-Net model creation."""
        from model import create_model
        
        model = create_model()
        assert model is not None
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(1, 3, 512, 512)
        output = model(dummy_input)
        
        assert output.shape == (1, 1, 512, 512)
        assert output.dtype == torch.float32
    
    def test_dataset_loading(self):
        """Test dataset loading functionality."""
        from dataset import AgriculturalSegmentationDataset
        
        # Create dummy dataset structure
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = os.path.join(temp_dir, 'images')
            masks_dir = os.path.join(temp_dir, 'masks')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            # Create dummy files (in practice, use real test images)
            # This is a simplified test
            
            dataset = AgriculturalSegmentationDataset(images_dir, masks_dir)
            assert len(dataset) == 0  # Empty dataset for this test
```

## 🎯 **Test Configuration**

### **Pytest Configuration**

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### **Coverage Configuration**

```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */venv/*
    */__pycache__/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
```

## 🚀 **Advanced Testing**

### **Performance Testing**

```python
# tests/performance/test_performance.py
import pytest
import time
import psutil
from src.dataset_combiner import DatasetCombiner

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.slow
    def test_memory_usage(self, test_config):
        """Test memory usage during processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        combiner = DatasetCombiner(test_config)
        results = combiner.combine_all_datasets()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 2GB for test dataset)
        assert memory_increase < 2 * 1024 * 1024 * 1024
    
    @pytest.mark.slow
    def test_processing_speed(self, test_config):
        """Test processing speed."""
        start_time = time.time()
        
        combiner = DatasetCombiner(test_config)
        results = combiner.combine_all_datasets()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Processing should complete within reasonable time (< 5 minutes for test)
        assert processing_time < 300
```

### **Stress Testing**

```python
# tests/stress/test_stress.py
import pytest
import tempfile
import shutil
import os
from src.dataset_combiner import DatasetCombiner

class TestStress:
    """Test system behavior under stress."""
    
    @pytest.mark.slow
    def test_large_dataset_handling(self, test_config):
        """Test handling of large datasets."""
        # Modify config for large dataset simulation
        test_config['processing']['batch_size'] = 1  # Small batches
        test_config['processing']['target_size'] = [1024, 1024]  # Large images
        
        combiner = DatasetCombiner(test_config)
        
        # This test would require a large test dataset
        # In practice, you'd create synthetic large datasets for testing
        pass
    
    @pytest.mark.slow
    def test_concurrent_processing(self, test_config):
        """Test concurrent processing capabilities."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def process_dataset():
            try:
                combiner = DatasetCombiner(test_config)
                results = combiner.combine_all_datasets()
                results_queue.put(('success', results))
            except Exception as e:
                results_queue.put(('error', str(e)))
        
        # Start multiple processing threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_dataset)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        results = []
        while not results_queue.empty():
            status, result = results_queue.get()
            results.append((status, result))
        
        # All threads should complete successfully
        assert len(results) == 3
        assert all(status == 'success' for status, _ in results)
```

## 🔧 **Test Utilities**

### **Shared Fixtures**

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
import os
import numpy as np
from PIL import Image

@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory for entire test session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_image(test_data_dir):
    """Create sample test image."""
    image_path = os.path.join(test_data_dir, 'test_image.png')
    image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    image.save(image_path)
    return image_path

@pytest.fixture
def sample_annotation(test_data_dir):
    """Create sample test annotation."""
    annotation_path = os.path.join(test_data_dir, 'test_annotation.png')
    annotation = Image.fromarray(np.random.randint(0, 2, (512, 512), dtype=np.uint8))
    annotation.save(annotation_path)
    return annotation_path

@pytest.fixture
def mock_dataset_structure(test_data_dir):
    """Create mock dataset directory structure."""
    structure = {
        'images': os.path.join(test_data_dir, 'images'),
        'annotations': os.path.join(test_data_dir, 'annotations')
    }
    
    for dir_path in structure.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return structure
```

### **Test Helpers**

```python
# tests/helpers.py
import os
import shutil
import tempfile
from contextlib import contextmanager

@contextmanager
def temporary_directory():
    """Context manager for temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

def create_test_image(path, size=(512, 512)):
    """Create a test image file."""
    import numpy as np
    from PIL import Image
    
    image = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    image.save(path)
    return path

def create_test_annotation(path, size=(512, 512)):
    """Create a test annotation file."""
    import numpy as np
    from PIL import Image
    
    annotation = Image.fromarray(np.random.randint(0, 2, size, dtype=np.uint8))
    annotation.save(path)
    return path

def assert_file_exists(file_path):
    """Assert that a file exists."""
    assert os.path.exists(file_path), f"File does not exist: {file_path}"

def assert_directory_exists(dir_path):
    """Assert that a directory exists."""
    assert os.path.exists(dir_path), f"Directory does not exist: {dir_path}"
    assert os.path.isdir(dir_path), f"Path is not a directory: {dir_path}"
```

## 📊 **Test Reporting**

### **Coverage Reports**

```bash
# Generate HTML coverage report
python -m pytest --cov=src --cov-report=html tests/

# Generate XML coverage report (for CI/CD)
python -m pytest --cov=src --cov-report=xml tests/

# Generate terminal coverage report
python -m pytest --cov=src --cov-report=term-missing tests/
```

### **Test Results**

```bash
# Generate HTML test report
python -m pytest --html=reports/test_report.html --self-contained-html tests/

# Generate JUnit XML report
python -m pytest --junitxml=reports/junit.xml tests/

# Run tests with verbose output
python -m pytest -v --tb=long tests/
```

## 🚨 **Common Testing Issues**

### **Issue 1: Import Errors**

**Problem**: `ModuleNotFoundError` when running tests

**Solution**:
```bash
# Install package in development mode
pip install -e .

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/project/root"

# Or use pytest with proper Python path
python -m pytest --import-mode=importlib tests/
```

### **Issue 2: Slow Tests**

**Problem**: Tests take too long to run

**Solution**:
```bash
# Run only fast tests
python -m pytest -m "not slow" tests/

# Run specific test categories
python -m pytest tests/unit/  # Only unit tests
python -m pytest tests/integration/  # Only integration tests

# Use pytest-xdist for parallel execution
python -m pytest -n auto tests/
```

### **Issue 3: Memory Issues**

**Problem**: Tests run out of memory

**Solution**:
```python
# Use smaller test datasets
@pytest.fixture
def small_test_config():
    return {
        'processing': {
            'target_size': [128, 128],  # Smaller images
            'batch_size': 1             # Smaller batches
        }
    }

# Clean up after tests
@pytest.fixture(autouse=True)
def cleanup():
    yield
    import gc
    gc.collect()
```

## 🎯 **Testing Best Practices**

### **1. Test Organization**

- **Group related tests** in test classes
- **Use descriptive test names** that explain the expected behavior
- **Keep tests independent** - no test should depend on another
- **Use fixtures** for common setup and teardown

### **2. Test Data Management**

- **Use temporary files** for test data
- **Clean up after tests** to avoid leaving artifacts
- **Create realistic test data** that mimics production scenarios
- **Use factories** to generate test data

### **3. Assertion Quality**

- **Test one thing per test** - avoid multiple assertions
- **Use specific assertions** rather than generic ones
- **Test edge cases** and error conditions
- **Validate both positive and negative scenarios**

### **4. Performance Considerations**

- **Mock external dependencies** to speed up tests
- **Use smaller datasets** for unit tests
- **Run slow tests separately** from fast tests
- **Profile tests** to identify bottlenecks

---

## 🎉 **Testing Summary**

### **What We've Covered**

✅ **Unit Testing**: Individual component testing  
✅ **Integration Testing**: Component interaction testing  
✅ **Performance Testing**: Speed and memory testing  
✅ **Stress Testing**: System behavior under load  
✅ **Test Configuration**: Pytest and coverage setup  
✅ **Test Utilities**: Shared fixtures and helpers  
✅ **Best Practices**: Testing guidelines and tips  

### **Next Steps**

1. **Run the test suite** to verify everything works
2. **Add tests** for new features and bug fixes
3. **Maintain test coverage** above 80%
4. **Integrate testing** into your CI/CD pipeline
5. **Use test results** to guide development decisions

---

**This testing guide provides a comprehensive foundation for testing the Agricultural Dataset Combination project. For specific testing scenarios, refer to the test examples and adapt them to your needs.** 🚀

---

<div align="center">

**Need More Details?** Check our [Complete Documentation](README.md) or [Source Code](https://github.com/selfishout/agricultural-dataset-combination)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/selfishout)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-torabi)

</div>
