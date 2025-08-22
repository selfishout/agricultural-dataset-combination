#!/usr/bin/env python3
"""
Test script to validate the sample segmentation project setup.
"""

import os
import sys
import importlib

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    required_modules = [
        'torch',
        'torchvision', 
        'numpy',
        'PIL',
        'matplotlib',
        'albumentations',
        'sklearn',
        'tqdm'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        print("   Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required modules imported successfully!")
        return True

def test_config():
    """Test configuration file."""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import config
        
        # Test dataset paths
        print("   Testing dataset paths...")
        config.validate_paths()
        
        print("   Testing directory creation...")
        config.create_directories()
        
        print("✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_dataset():
    """Test dataset loader."""
    print("\n📁 Testing dataset loader...")
    
    try:
        from dataset import create_data_loaders
        
        # Create data loaders with minimal batch size
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=2, num_workers=0
        )
        
        print(f"   ✅ Data loaders created:")
        print(f"      Training: {len(train_loader)} batches")
        print(f"      Validation: {len(val_loader)} batches")
        print(f"      Testing: {len(test_loader)} batches")
        
        # Test loading a batch
        print("   Testing batch loading...")
        batch = next(iter(train_loader))
        print(f"      Batch keys: {list(batch.keys())}")
        print(f"      Image shape: {batch['image'].shape}")
        print(f"      Mask shape: {batch['mask'].shape}")
        
        print("✅ Dataset test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset test failed: {e}")
        return False

def test_model():
    """Test model creation."""
    print("\n🧠 Testing model...")
    
    try:
        import torch
        from model import create_model, create_loss_function, create_optimizer
        
        # Create model
        model = create_model()
        
        # Create loss and optimizer
        criterion = create_loss_function()
        optimizer = create_optimizer(model)
        
        print(f"   ✅ Model created successfully")
        print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("   Testing forward pass...")
        dummy_input = torch.randn(1, 3, 512, 512)
        output = model(dummy_input)
        print(f"      Input shape: {dummy_input.shape}")
        print(f"      Output shape: {output.shape}")
        
        print("✅ Model test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False

def test_training_components():
    """Test training components."""
    print("\n🚀 Testing training components...")
    
    try:
        from train_segmentation import Trainer
        
        print("   ✅ Training components imported successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Training components test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Sample Segmentation Project Setup Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_dataset,
        test_model,
        test_training_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run training: python train_segmentation.py")
        print("   2. Evaluate model: python evaluate_model.py")
        print("   3. View results in the 'results/' directory")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Check dataset paths in config.py")
        print("   3. Ensure all required files exist")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
