# 🚀 Quick Start Guide

Get up and running with the Agricultural Dataset Combination project in minutes! This guide will walk you through the essential steps to start combining agricultural datasets and training segmentation models.

## ⚡ **5-Minute Quick Start**

### **Step 1: Basic Setup**
```bash
# Clone and setup (if not already done)
git clone https://github.com/selfishout/agricultural-dataset-combination.git
cd agricultural-dataset-combination

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Test Installation**
```bash
# Verify everything works
python -c "import src; print('✅ Setup complete!')"

# Test sample project
cd sample_segmentation_project
python test_setup.py
```

### **Step 3: Run Sample Project**
```bash
# Start training (uses demo data)
python train_segmentation.py

# Monitor progress
tensorboard --logdir logs
```

**🎉 You're ready to go!** Continue reading for detailed workflows.

## 🌾 **Dataset Combination Workflow**

### **1. Prepare Your Datasets**

Create the following directory structure:
```
source_datasets/
├── phenobench/           # PhenoBench dataset
├── capsicum/             # Capsicum Annuum dataset
├── vineyard/             # Vineyard dataset
└── weed_augmented/       # Weed augmented dataset
```

### **2. Configure the Project**

Edit `config/dataset_config.yaml`:
```yaml
storage:
  output_dir: "/path/to/your/output/directory"
  source_dir: "/path/to/your/source_datasets"

processing:
  target_size: [512, 512]
  image_format: "PNG"
  augmentation: true
```

### **3. Combine Datasets**

```bash
# Run the complete combination pipeline
python scripts/combine_datasets.py

# Or run step by step:
python scripts/setup_datasets.py
python scripts/combine_datasets.py
python scripts/validate_combination.py
```

### **4. Verify Results**

Check your combined dataset:
```bash
# View dataset statistics
ls -la /path/to/your/output/directory/

# Check metadata
cat /path/to/your/output/directory/metadata.json
```

## 🎨 **Sample Segmentation Project**

### **Quick Training**

```bash
cd sample_segmentation_project

# Update paths in config.py
# Then start training:
python train_segmentation.py
```

### **Monitor Training**

```bash
# Start TensorBoard
tensorboard --logdir logs

# Open http://localhost:6006 in your browser
```

### **Evaluate Results**

```bash
# Evaluate trained model
python evaluate_model.py

# View results in results/ directory
```

## 📊 **Expected Results**

### **Dataset Combination**
- **Processing Time**: 2-4 hours for full pipeline
- **Output Size**: ~50GB combined dataset
- **Image Count**: 62,763 total images
- **Format**: PNG (512×512 pixels)

### **Training Performance**
- **Training Time**: 2-4 hours for 10 epochs
- **Memory Usage**: ~8GB with batch size 8
- **Expected Dice Score**: 0.7+ after training
- **Model Size**: 31M+ parameters

## 🔧 **Common Quick Configurations**

### **For Development/Testing**
```yaml
# config/dataset_config.yaml
processing:
  target_size: [256, 256]  # Smaller for faster processing
  batch_size: 4            # Reduce memory usage
  quality_check: false     # Skip quality checks for speed
```

### **For Production**
```yaml
# config/dataset_config.yaml
processing:
  target_size: [1024, 1024]  # Higher resolution
  batch_size: 16             # Larger batches
  quality_check: true         # Full quality assurance
  backup_original: true       # Keep original data
```

### **For GPU Training**
```python
# sample_segmentation_project/config.py
DEVICE = "cuda"
BATCH_SIZE = 16
NUM_WORKERS = 8
```

## 🚨 **Quick Troubleshooting**

### **Common Issues & Solutions**

#### **Issue: "No module named 'torch'"**
```bash
# Install PyTorch
pip install torch torchvision

# Or for GPU support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **Issue: "Out of memory"**
```bash
# Reduce batch size
# Edit config.py: BATCH_SIZE = 4

# Or reduce image size
# Edit dataset_config.yaml: target_size: [256, 256]
```

#### **Issue: "Dataset not found"**
```bash
# Check paths in config files
# Ensure source datasets exist
# Verify directory permissions
```

#### **Issue: "CUDA not available"**
```bash
# Install CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or check CUDA installation
nvidia-smi
```

## 📈 **Performance Optimization**

### **Speed Up Processing**
```bash
# Use multiple cores
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Reduce image quality checks
# Edit config: quality_check: false
```

### **Reduce Memory Usage**
```yaml
# config/dataset_config.yaml
processing:
  batch_size: 4
  target_size: [256, 256]
  compression: true
```

### **Optimize Training**
```python
# sample_segmentation_project/config.py
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = False  # For CPU training
```

## 🔍 **Quick Validation**

### **Check Dataset Quality**
```bash
# Run validation script
python scripts/validate_combination.py

# Expected output:
# ✅ Total images: 62,763
# ✅ Training split: 43,934
# ✅ Validation split: 12,552
# ✅ Test split: 6,277
```

### **Test Data Loading**
```python
# Quick test script
from sample_segmentation_project.dataset import create_data_loaders

train_loader, val_loader, test_loader = create_data_loaders(batch_size=2)

# Test first batch
for batch in train_loader:
    print(f"Image shape: {batch['image'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    break
```

### **Verify Model Creation**
```python
# Test model creation
from sample_segmentation_project.model import create_model

model = create_model()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 🎯 **Next Steps After Quick Start**

### **Immediate Actions**
1. **Explore the combined dataset** structure and contents
2. **Run a few training epochs** to verify everything works
3. **Check TensorBoard** for training progress
4. **Validate results** with sample predictions

### **Short-term Goals**
1. **Customize configuration** for your specific needs
2. **Optimize parameters** based on your hardware
3. **Add your own datasets** to the combination pipeline
4. **Extend the model** for your specific use case

### **Long-term Development**
1. **Contribute improvements** to the project
2. **Share your results** with the community
3. **Publish research** using the combined dataset
4. **Build applications** on top of the framework

## 📚 **Quick Reference Commands**

### **Essential Commands**
```bash
# Dataset combination
python scripts/combine_datasets.py

# Training
cd sample_segmentation_project
python train_segmentation.py

# Monitoring
tensorboard --logdir logs

# Evaluation
python evaluate_model.py

# Testing
python test_setup.py
```

### **Configuration Files**
- **Main config**: `config/dataset_config.yaml`
- **Sample project**: `sample_segmentation_project/config.py`
- **Requirements**: `requirements.txt`
- **Development**: `requirements-dev.txt`

### **Key Directories**
- **Source code**: `src/`
- **Scripts**: `scripts/`
- **Sample project**: `sample_segmentation_project/`
- **Tests**: `tests/`
- **Documentation**: `docs/`

## 🆘 **Quick Help**

### **When You Need Help**
1. **Check this guide** for common solutions
2. **Read the [Troubleshooting Guide](troubleshooting.md)**
3. **Search [GitHub Issues](https://github.com/selfishout/agricultural-dataset-combination/issues)**
4. **Ask the [Community](https://github.com/selfishout/agricultural-dataset-combination/discussions)**

### **Useful Resources**
- **Full Documentation**: [docs/README.md](README.md)
- **API Reference**: [docs/api_reference.md](api_reference.md)
- **Contributing Guide**: [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Project README**: [README.md](../README.md)

---

## 🎉 **You're All Set!**

Congratulations! You've completed the quick start and are ready to:

✅ **Combine agricultural datasets**  
✅ **Train segmentation models**  
✅ **Monitor training progress**  
✅ **Evaluate model performance**  
✅ **Customize for your needs**  

**Your agricultural AI journey is underway!** 🌾🤖

---

<div align="center">

**Ready for More?** Explore our [Complete Documentation](README.md) or [Join the Community](https://github.com/selfishout/agricultural-dataset-combination/discussions)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/selfishout)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-torabi)

</div>
