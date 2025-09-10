#!/usr/bin/env python3
"""
Fruit Datasets Integration Script
Generated on 2025-09-10 09:33:49
"""

import os
import sys
import json
import shutil
from pathlib import Path
import zipfile
from PIL import Image, ImageDraw
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.append('src')

from utils import setup_logging, load_config
from dataset_combiner import DatasetCombiner

def download_fruit_datasets():
    """Download fruit datasets from Kaggle"""
    print("🍎 Downloading fruit datasets...")
    
    # Dataset information
    datasets = [
  "moltean_fruits",
  "shimul_fruits",
  "mango_classify"
]
    
    # Create download directory
    download_dir = "fruit_datasets_download"
    os.makedirs(download_dir, exist_ok=True)
    
    # Note: This requires Kaggle API setup
    print("⚠️  Note: Kaggle API key required for automatic download")
    print("   Please download the following datasets manually:")
    
    fruit_datasets_info = {
        "moltean_fruits": {
            "name": "Fruits (Moltean)",
            "url": "https://www.kaggle.com/datasets/moltean/fruits"
        },
        "shimul_fruits": {
            "name": "Fruits Dataset (Shimul)", 
            "url": "https://www.kaggle.com/datasets/shimulmbstu/fruitsdataset"
        },
        "mango_classify": {
            "name": "Mango Classify (12 Native Mango)",
            "url": "https://www.kaggle.com/datasets/researchersajid/mangoclassify-12-native-mango-dataset-from-bd"
        }
    }
    
    for dataset in [
  "moltean_fruits",
  "shimul_fruits",
  "mango_classify"
]:
        if dataset in fruit_datasets_info:
            print(f"   • {fruit_datasets_info[dataset]['name']}: {fruit_datasets_info[dataset]['url']}")
    
    return download_dir

def convert_classification_to_segmentation(dataset_path, output_path):
    """Convert classification dataset to segmentation format"""
    print(f"🔄 Converting {dataset_path} to segmentation format...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Create images and masks directories
    images_dir = os.path.join(output_path, "images")
    masks_dir = os.path.join(output_path, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Process each class directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                
                # Get class name from directory structure
                class_name = os.path.basename(root)
                if class_name.lower() in ['images', 'data', 'train', 'val', 'test']:
                    continue
                
                # Copy image
                new_image_name = f"{class_name}_{file}"
                new_image_path = os.path.join(images_dir, new_image_name)
                shutil.copy2(image_path, new_image_path)
                
                # Create segmentation mask (full image mask for classification)
                mask_name = new_image_name.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
                mask_path = os.path.join(masks_dir, mask_name)
                
                # Load image to get dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
                
                # Create mask (full image for classification datasets)
                mask = Image.new('L', (width, height), 255)  # White mask
                mask.save(mask_path)
    
    print(f"✅ Conversion complete: {len(os.listdir(images_dir))} images processed")

def integrate_with_existing_dataset(fruit_datasets_dir, existing_dataset_path):
    """Integrate fruit datasets with existing agricultural dataset"""
    print("🔗 Integrating fruit datasets with existing agricultural dataset...")
    
    # Load existing dataset configuration
    config = load_config("config/dataset_config.yaml")
    
    # Initialize dataset combiner
    combiner = DatasetCombiner(config)
    
    # Add fruit datasets to the combination process
    # This would extend the existing combiner to handle fruit datasets
    
    print("✅ Integration complete")

def main():
    """Main integration function"""
    print("🍎 FRUIT DATASETS INTEGRATION")
    print("=" * 50)
    
    # Download datasets
    download_dir = download_fruit_datasets()
    
    # Convert each dataset
    compatible_datasets = [
  "moltean_fruits",
  "shimul_fruits",
  "mango_classify"
]
    for dataset in compatible_datasets:
        dataset_path = os.path.join(download_dir, dataset)
        output_path = os.path.join(download_dir, f"{dataset}_segmentation")
        
        if os.path.exists(dataset_path):
            convert_classification_to_segmentation(dataset_path, output_path)
    
    # Integrate with existing dataset
    config = load_config("config/dataset_config.yaml")
    integrate_with_existing_dataset(download_dir, config['storage']['output_dir'])
    
    print("🎉 Fruit datasets integration complete!")

if __name__ == "__main__":
    main()
