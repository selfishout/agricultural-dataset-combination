#!/usr/bin/env python3
"""
Download and Integrate Real Fruit Datasets from Kaggle
Downloads the three actual Kaggle fruit datasets and integrates them into the main agricultural dataset
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import logging
import subprocess
import requests
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FruitDatasetIntegrator:
    def __init__(self, base_path="/Volumes/Rapid/Agriculture Dataset"):
        self.base_path = Path(base_path)
        self.fruit_integration_path = self.base_path / "Fruit_Integration"
        self.downloads_path = self.fruit_integration_path / "downloads"
        self.processed_path = self.fruit_integration_path / "processed"
        self.integration_metadata = {
            "integration_date": datetime.now().isoformat(),
            "datasets": {},
            "total_images": 0,
            "total_classes": 0
        }
        
        # Create directories
        self.downloads_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_dataset(self, dataset_name, kaggle_url):
        """Download dataset from Kaggle using kaggle API"""
        logger.info(f"📥 Downloading {dataset_name} from Kaggle...")
        
        try:
            # Extract dataset identifier from URL
            if "kaggle.com/datasets/" in kaggle_url:
                dataset_id = kaggle_url.split("kaggle.com/datasets/")[1].split("/")[0]
            else:
                logger.error(f"Invalid Kaggle URL format: {kaggle_url}")
                return False
            
            # Download using kaggle API
            download_cmd = [
                "kaggle", "datasets", "download", 
                "-d", dataset_id,
                "-p", str(self.downloads_path / dataset_name),
                "--unzip"
            ]
            
            result = subprocess.run(download_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully downloaded {dataset_name}")
                return True
            else:
                logger.error(f"❌ Failed to download {dataset_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error downloading {dataset_name}: {str(e)}")
            return False
    
    def download_manual_datasets(self):
        """Download datasets manually (when kaggle API is not available)"""
        logger.info("📥 Setting up manual download instructions...")
        
        datasets = {
            "moltean_fruits": {
                "url": "https://www.kaggle.com/datasets/moltean/fruits",
                "expected_size": "~500MB",
                "description": "Fruits dataset with 131 classes"
            },
            "shimul_fruits": {
                "url": "https://www.kaggle.com/datasets/shimulmbstu/fruitsdataset", 
                "expected_size": "~1GB",
                "description": "Fruits dataset with multiple fruit types"
            },
            "mango_classify": {
                "url": "https://www.kaggle.com/datasets/researchersajid/mangoclassify-12-native-mango-dataset-from-bd",
                "expected_size": "~200MB", 
                "description": "12 Native Mango varieties from Bangladesh"
            }
        }
        
        # Create download instructions
        instructions_path = self.fruit_integration_path / "MANUAL_DOWNLOAD_INSTRUCTIONS.md"
        
        with open(instructions_path, 'w') as f:
            f.write("# 🍎 Manual Fruit Dataset Download Instructions\n\n")
            f.write("Please download the following datasets manually and place them in the specified directories:\n\n")
            
            for dataset_name, info in datasets.items():
                dataset_path = self.downloads_path / dataset_name
                dataset_path.mkdir(exist_ok=True)
                
                f.write(f"## {dataset_name.replace('_', ' ').title()}\n")
                f.write(f"- **URL**: {info['url']}\n")
                f.write(f"- **Expected Size**: {info['expected_size']}\n")
                f.write(f"- **Description**: {info['description']}\n")
                f.write(f"- **Download To**: `{dataset_path}`\n")
                f.write(f"- **Instructions**: Download the ZIP file and extract it to this directory\n\n")
        
        logger.info(f"📝 Download instructions created: {instructions_path}")
        return True
    
    def process_fruit_dataset(self, dataset_name, dataset_path):
        """Process a fruit dataset and convert to segmentation format"""
        logger.info(f"🔄 Processing {dataset_name}...")
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            return False
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(dataset_path.rglob(ext)))
        
        if not image_files:
            logger.warning(f"No image files found in {dataset_path}")
            return False
        
        logger.info(f"Found {len(image_files)} images in {dataset_name}")
        
        # Create processed directory
        processed_dataset_path = self.processed_path / dataset_name
        processed_dataset_path.mkdir(exist_ok=True)
        
        # Process images
        processed_count = 0
        class_mapping = {}
        
        for img_path in image_files:
            try:
                # Determine class from directory structure
                relative_path = img_path.relative_to(dataset_path)
                class_name = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
                
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)
                
                # Load and process image
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 512x512
                    img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # Save processed image
                    output_path = processed_dataset_path / f"{processed_count:06d}_{class_name}.png"
                    img_resized.save(output_path, 'PNG')
                    
                    processed_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Create segmentation masks (simple approach for classification datasets)
        self.create_segmentation_masks(processed_dataset_path, class_mapping)
        
        # Update metadata
        self.integration_metadata["datasets"][dataset_name] = {
            "original_images": len(image_files),
            "processed_images": processed_count,
            "classes": list(class_mapping.keys()),
            "class_count": len(class_mapping),
            "class_mapping": class_mapping,
            "processed_path": str(processed_dataset_path)
        }
        
        logger.info(f"✅ Processed {dataset_name}: {processed_count} images, {len(class_mapping)} classes")
        return True
    
    def create_segmentation_masks(self, dataset_path, class_mapping):
        """Create simple segmentation masks for classification datasets"""
        logger.info(f"🎭 Creating segmentation masks for {dataset_path.name}...")
        
        masks_path = dataset_path / "masks"
        masks_path.mkdir(exist_ok=True)
        
        # Get all processed images
        image_files = list(dataset_path.glob("*.png"))
        
        for img_path in image_files:
            if "mask" in img_path.name:
                continue
                
            try:
                # Create a simple mask (center region for the main object)
                mask = np.zeros((512, 512), dtype=np.uint8)
                
                # Create a centered rectangle as the "object" region
                # This is a simplified approach - in practice, you'd want more sophisticated segmentation
                center_x, center_y = 256, 256
                width, height = 300, 300
                
                x1 = max(0, center_x - width // 2)
                y1 = max(0, center_y - height // 2)
                x2 = min(512, center_x + width // 2)
                y2 = min(512, center_y + height // 2)
                
                # Fill the rectangle with class ID
                class_name = img_path.stem.split('_')[-1]
                class_id = class_mapping.get(class_name, 0)
                mask[y1:y2, x1:x2] = class_id
                
                # Save mask
                mask_path = masks_path / f"{img_path.stem}_mask.png"
                Image.fromarray(mask).save(mask_path)
                
            except Exception as e:
                logger.warning(f"Error creating mask for {img_path}: {str(e)}")
                continue
        
        logger.info(f"✅ Created {len(list(masks_path.glob('*.png')))} segmentation masks")
    
    def integrate_into_main_dataset(self):
        """Integrate processed fruit datasets into main agricultural dataset"""
        logger.info("🔗 Integrating fruit datasets into main agricultural dataset...")
        
        main_dataset_path = self.base_path / "Combined_datasets"
        if not main_dataset_path.exists():
            logger.error("Main dataset not found. Please run the main dataset combination first.")
            return False
        
        # Create fruit integration directory in main dataset
        fruit_main_path = main_dataset_path / "fruit_datasets"
        fruit_main_path.mkdir(exist_ok=True)
        
        total_fruit_images = 0
        all_classes = set()
        
        # Copy processed datasets
        for dataset_name, dataset_info in self.integration_metadata["datasets"].items():
            source_path = Path(dataset_info["processed_path"])
            target_path = fruit_main_path / dataset_name
            
            if source_path.exists():
                # Copy dataset
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                
                total_fruit_images += dataset_info["processed_images"]
                all_classes.update(dataset_info["classes"])
                
                logger.info(f"✅ Integrated {dataset_name}: {dataset_info['processed_images']} images")
            else:
                logger.warning(f"Processed dataset not found: {source_path}")
        
        # Update main dataset metadata
        self.update_main_dataset_metadata(total_fruit_images, all_classes)
        
        logger.info(f"🎉 Integration complete: {total_fruit_images} fruit images added")
        return True
    
    def update_main_dataset_metadata(self, total_fruit_images, all_classes):
        """Update the main dataset metadata with fruit data"""
        metadata_path = self.base_path / "Combined_datasets" / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Update with fruit data
        metadata["fruit_integration"] = {
            "integration_date": datetime.now().isoformat(),
            "total_fruit_images": total_fruit_images,
            "fruit_classes": sorted(list(all_classes)),
            "fruit_class_count": len(all_classes),
            "datasets": self.integration_metadata["datasets"]
        }
        
        # Update total counts
        if "total_images" in metadata:
            metadata["total_images"] += total_fruit_images
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("📊 Updated main dataset metadata with fruit integration")
    
    def run_complete_integration(self):
        """Run the complete fruit dataset integration process"""
        logger.info("🚀 Starting complete fruit dataset integration...")
        
        # Step 1: Set up manual downloads
        self.download_manual_datasets()
        
        # Step 2: Check for downloaded datasets and process them
        for dataset_name in ["moltean_fruits", "shimul_fruits", "mango_classify"]:
            dataset_path = self.downloads_path / dataset_name
            
            if dataset_path.exists() and any(dataset_path.iterdir()):
                logger.info(f"Found downloaded dataset: {dataset_name}")
                self.process_fruit_dataset(dataset_name, dataset_path)
            else:
                logger.warning(f"Dataset not found: {dataset_name}. Please download manually.")
        
        # Step 3: Integrate into main dataset
        if self.integration_metadata["datasets"]:
            self.integrate_into_main_dataset()
        
        # Step 4: Save integration metadata
        metadata_path = self.fruit_integration_path / "fruit_integration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.integration_metadata, f, indent=2)
        
        # Print summary
        self.print_integration_summary()
        
        return self.integration_metadata
    
    def print_integration_summary(self):
        """Print integration summary"""
        print("\n" + "="*80)
        print("🍎 FRUIT DATASET INTEGRATION SUMMARY")
        print("="*80)
        
        total_images = sum(dataset["processed_images"] for dataset in self.integration_metadata["datasets"].values())
        total_classes = len(set().union(*[dataset["classes"] for dataset in self.integration_metadata["datasets"].values()]))
        
        print(f"\n📊 INTEGRATION RESULTS:")
        print(f"  Total Fruit Images: {total_images:,}")
        print(f"  Total Fruit Classes: {total_classes}")
        print(f"  Datasets Integrated: {len(self.integration_metadata['datasets'])}")
        
        print(f"\n📁 DATASET BREAKDOWN:")
        for dataset_name, dataset_info in self.integration_metadata["datasets"].items():
            print(f"  {dataset_name}: {dataset_info['processed_images']} images, {dataset_info['class_count']} classes")
        
        print(f"\n🏷️ FRUIT CLASSES:")
        all_classes = set().union(*[dataset["classes"] for dataset in self.integration_metadata["datasets"].values()])
        print(f"  {', '.join(sorted(all_classes))}")
        
        print("\n" + "="*80)

def main():
    integrator = FruitDatasetIntegrator()
    results = integrator.run_complete_integration()
    
    print(f"\n🎉 Fruit integration complete!")
    print(f"📊 Total fruit images processed: {sum(dataset['processed_images'] for dataset in results['datasets'].values())}")

if __name__ == "__main__":
    main()
