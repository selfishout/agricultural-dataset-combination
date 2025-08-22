# 🚀 Installation Guide

Complete installation guide for the Agricultural Dataset Combination project. This guide covers all installation methods and requirements.

## 📋 **Prerequisites**

### **System Requirements**
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 or higher (3.9+ recommended)
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 100GB+ free space for dataset processing
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but recommended for training (CUDA 11.0+)

### **Software Dependencies**
- **Git**: For cloning the repository
- **Python**: Core runtime environment
- **pip**: Python package manager
- **virtualenv/venv**: Python virtual environment tool

## 🔧 **Installation Methods**

### **Method 1: From Source (Recommended)**

#### **Step 1: Clone the Repository**
```bash
# Clone the main repository
git clone https://github.com/selfishout/agricultural-dataset-combination.git
cd agricultural-dataset-combination

# Or clone a specific branch
git clone -b develop https://github.com/selfishout/agricultural-dataset-combination.git
```

#### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### **Step 3: Install Dependencies**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

#### **Step 4: Verify Installation**
```bash
# Test basic functionality
python -c "import src; print('Installation successful!')"

# Run tests
python -m pytest tests/ -v
```

### **Method 2: From PyPI (Stable Release)**

```bash
# Install from PyPI
pip install agricultural-dataset-combination

# Or install with specific version
pip install agricultural-dataset-combination==1.0.0
```

### **Method 3: Using Conda (Alternative)**

```bash
# Create conda environment
conda create -n agri-dataset python=3.9

# Activate environment
conda activate agri-dataset

# Install dependencies
conda install -c conda-forge pytorch torchvision opencv
pip install -r requirements.txt
```

## 📦 **Dependency Installation Details**

### **Core Dependencies**
```bash
# Data processing
pip install numpy pandas Pillow opencv-python scipy

# Machine learning
pip install torch torchvision scikit-learn

# Data augmentation
pip install albumentations

# Visualization
pip install matplotlib seaborn plotly

# Utilities
pip install tqdm PyYAML
```

### **Optional Dependencies**
```bash
# GPU support (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Jupyter notebooks
pip install jupyter jupyterlab ipykernel

# Development tools
pip install pytest flake8 black mypy

# Documentation
pip install sphinx sphinx-rtd-theme
```

### **System-Specific Dependencies**

#### **Ubuntu/Debian**
```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```

#### **macOS**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system packages
brew install python3 opencv
```

#### **Windows**
```bash
# Install Visual C++ Redistributable
# Download from Microsoft's website

# Install Windows Subsystem for Linux (WSL) for better performance
wsl --install
```

## 🔍 **Installation Verification**

### **Basic Functionality Test**
```python
# Test core imports
import src
from src.dataset_combiner import DatasetCombiner
from src.preprocessing import ImagePreprocessor
from src.visualization import DatasetVisualizer

print("✅ Core modules imported successfully")
```

### **Configuration Test**
```python
# Test configuration loading
from config.dataset_config import load_config

config = load_config('config/dataset_config.yaml')
print("✅ Configuration loaded successfully")
print(f"Output directory: {config['storage']['output_dir']}")
```

### **Sample Project Test**
```bash
# Navigate to sample project
cd sample_segmentation_project

# Test setup
python test_setup.py

# Expected output: All tests passed
```

## 🚨 **Common Installation Issues**

### **Issue 1: Python Version Compatibility**
```bash
# Check Python version
python --version

# If version < 3.8, install newer version
# On Ubuntu:
sudo apt-get install python3.9 python3.9-venv

# On macOS:
brew install python@3.9
```

### **Issue 2: Missing System Libraries**
```bash
# On Ubuntu/Debian
sudo apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# On CentOS/RHEL
sudo yum install -y \
    mesa-libGL \
    mesa-libEGL \
    libXext \
    libXrender
```

### **Issue 3: PyTorch Installation Issues**
```bash
# Clear pip cache
pip cache purge

# Install PyTorch with specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### **Issue 4: OpenCV Installation Problems**
```bash
# Try alternative OpenCV package
pip uninstall opencv-python
pip install opencv-python-headless

# Or install from conda
conda install -c conda-forge opencv
```

### **Issue 5: Memory Issues**
```bash
# Reduce batch size in configuration
# Edit config/dataset_config.yaml
processing:
  batch_size: 4  # Reduce from default 8

# Or use smaller test datasets first
```

## 🔧 **Post-Installation Setup**

### **Environment Variables**
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export AGRICULTURAL_DATASET_PATH="/path/to/your/datasets"
export PYTHONPATH="${PYTHONPATH}:/path/to/agricultural-dataset-combination"

# Reload shell profile
source ~/.bashrc  # or ~/.zshrc
```

### **Configuration Setup**
```yaml
# Edit config/dataset_config.yaml
storage:
  output_dir: "/path/to/your/output/directory"
  source_dir: "/path/to/your/source_datasets"
  intermediate_dir: "/path/to/intermediate/files"

processing:
  target_size: [512, 512]
  image_format: "PNG"
  augmentation: true
  quality_check: true
```

### **Dataset Preparation**
```bash
# Create dataset directory structure
mkdir -p /path/to/your/source_datasets/{phenobench,capsicum,vineyard}

# Download source datasets to respective directories
# (Follow individual dataset instructions)
```

## 📊 **Installation Performance**

### **Installation Time Estimates**
| Component | Time Estimate | Dependencies |
|-----------|---------------|--------------|
| **Core Installation** | 2-5 minutes | Python, pip |
| **Dependencies** | 5-15 minutes | PyTorch, OpenCV |
| **GPU Support** | +2-5 minutes | CUDA toolkit |
| **Development Tools** | +3-5 minutes | Testing, linting |
| **Total Time** | 10-30 minutes | All components |

### **Storage Requirements**
| Component | Size | Description |
|-----------|------|-------------|
| **Source Code** | ~50MB | Repository and dependencies |
| **Virtual Environment** | ~2GB | Python packages |
| **Sample Datasets** | ~1GB | Test data |
| **Total Installation** | ~3GB | Complete setup |

## 🧪 **Testing Your Installation**

### **Run Complete Test Suite**
```bash
# Run all tests
python -m pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/test_dataset_combiner.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_visualization.py -v
```

### **Test Sample Project**
```bash
# Navigate to sample project
cd sample_segmentation_project

# Run setup test
python test_setup.py

# Expected: All 5 tests passed
```

### **Test Dataset Processing**
```bash
# Test with sample data
python scripts/setup_datasets.py --help
python scripts/combine_datasets.py --help
python scripts/validate_combination.py --help
```

## 🔄 **Updating Installation**

### **Update from Source**
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall in development mode
pip install -e . --force-reinstall
```

### **Update from PyPI**
```bash
# Update to latest version
pip install --upgrade agricultural-dataset-combination

# Or install specific version
pip install agricultural-dataset-combination==1.1.0
```

## 🆘 **Getting Help**

### **Installation Support**
- **GitHub Issues**: [Report installation problems](https://github.com/selfishout/agricultural-dataset-combination/issues)
- **GitHub Discussions**: [Ask for help](https://github.com/selfishout/agricultural-dataset-combination/discussions)
- **Documentation**: Check this guide and other docs
- **Troubleshooting**: See [troubleshooting guide](troubleshooting.md)

### **Community Resources**
- **Stack Overflow**: Tag questions with `agricultural-dataset-combination`
- **Reddit**: r/MachineLearning, r/ComputerVision
- **Discord**: Agricultural AI community servers

---

## 🎉 **Installation Complete!**

Congratulations! You've successfully installed the Agricultural Dataset Combination project. 

**Next Steps:**
1. **Read the [Quick Start Guide](quick_start.md)** to get started
2. **Explore the [Configuration Guide](configuration.md)** for customization
3. **Try the [Sample Project](sample_project.md)** to see it in action
4. **Join the [Community](https://github.com/selfishout/agricultural-dataset-combination/discussions)** for support

**Your agricultural AI journey begins now!** 🌾🤖

---

<div align="center">

**Need Help?** Check our [Troubleshooting Guide](troubleshooting.md) or [Ask the Community](https://github.com/selfishout/agricultural-dataset-combination/discussions)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/selfishout)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-torabi)

</div>
