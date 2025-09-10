#!/usr/bin/env python3
"""
Test script for fruit datasets integration
Creates a small test dataset to verify the integration process
"""

import os
import sys
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

def create_test_fruit_dataset():
    """Create a small test fruit dataset for integration testing"""
    print("🧪 Creating test fruit dataset...")
    
    # Create test dataset directory
    test_dir = "test_fruit_dataset"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create class directories
    classes = ["apple", "banana", "orange"]
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 5 test images per class
        for i in range(5):
            # Create a simple colored image
            img = Image.new('RGB', (256, 256), color=(
                np.random.randint(100, 255),  # R
                np.random.randint(100, 255),  # G
                np.random.randint(100, 255)   # B
            ))
            
            # Add some text to identify the class
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"{class_name}_{i}", fill=(0, 0, 0))
            
            # Save image
            img_path = os.path.join(class_dir, f"{class_name}_{i}.jpg")
            img.save(img_path)
    
    print(f"✅ Test dataset created: {test_dir}")
    print(f"   Classes: {classes}")
    print(f"   Images per class: 5")
    print(f"   Total images: {len(classes) * 5}")
    
    return test_dir

def test_integration_process():
    """Test the integration process with the test dataset"""
    print("\n🔄 Testing integration process...")
    
    # Import the integrator
    from integrate_fruit_datasets_comprehensive import FruitDatasetIntegrator
    
    # Create test dataset
    test_dataset_path = create_test_fruit_dataset()
    
    # Initialize integrator
    integrator = FruitDatasetIntegrator()
    
    # Setup directories
    integrator.setup_directories()
    
    # Copy test dataset to download directory
    test_download_path = os.path.join(integrator.download_dir, "test_fruits")
    if os.path.exists(test_download_path):
        shutil.rmtree(test_download_path)
    shutil.copytree(test_dataset_path, test_download_path)
    
    # Add test dataset to integrator's fruit datasets
    integrator.fruit_datasets["test_fruits"] = {
        "name": "Test Fruits",
        "url": "test",
        "kaggle_id": "test",
        "estimated_images": 15,
        "estimated_classes": 3,
        "description": "Test fruit dataset for integration testing"
    }
    
    # Process the test dataset
    print("\n📊 Processing test dataset...")
    result = integrator.process_fruit_dataset("test_fruits", test_download_path)
    
    if result:
        print("✅ Test processing successful!")
        print(f"   Processed images: {result['metadata']['processed_images']}")
        print(f"   Classes: {result['metadata']['classes']}")
        print(f"   Output directory: {result['output_dir']}")
        
        # Verify output structure
        images_dir = result['images_dir']
        masks_dir = result['masks_dir']
        
        image_files = os.listdir(images_dir)
        mask_files = os.listdir(masks_dir)
        
        print(f"   Generated images: {len(image_files)}")
        print(f"   Generated masks: {len(mask_files)}")
        
        # Check if images are properly resized
        if image_files:
            sample_img = Image.open(os.path.join(images_dir, image_files[0]))
            print(f"   Image size: {sample_img.size}")
            
        if mask_files:
            sample_mask = Image.open(os.path.join(masks_dir, mask_files[0]))
            print(f"   Mask size: {sample_mask.size}")
        
        return True
    else:
        print("❌ Test processing failed!")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    test_dirs = ["test_fruit_dataset", "fruit_integration"]
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   Removed: {dir_path}")

def main():
    """Main test function"""
    print("🧪 FRUIT DATASETS INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Test the integration process
        success = test_integration_process()
        
        if success:
            print("\n✅ All tests passed! Integration process is working correctly.")
            print("\n📋 Next steps:")
            print("1. Download the actual fruit datasets from Kaggle")
            print("2. Place them in fruit_integration/downloads/")
            print("3. Run: python3 integrate_fruit_datasets_comprehensive.py --process")
        else:
            print("\n❌ Tests failed! Please check the integration process.")
        
        # Ask if user wants to keep test files
        response = input("\n🧹 Clean up test files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_test_files()
        else:
            print("📁 Test files kept for inspection.")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

if __name__ == "__main__":
    main()
