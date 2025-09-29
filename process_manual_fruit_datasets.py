#!/usr/bin/env python3
"""
Process Manually Downloaded Fruit Datasets
Handles decompression and integration of the three fruit datasets
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image
import numpy as np
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManualFruitDatasetProcessor:
    def __init__(self, base_path="/Volumes/Rapid/Agriculture Dataset"):
        self.base_path = Path(base_path)
        self.fruit_integration_path = self.base_path / "Fruit_Integration"
        self.processed_path = self.fruit_integration_path / "processed"
        self.integration_metadata = {
            "integration_date": datetime.now().isoformat(),
            "datasets": {},
            "total_images": 0,
            "total_classes": 0
        }
        
        # Create directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def find_fruit_datasets(self):
        """Find the three fruit datasets in the base directory"""
        logger.info("🔍 Searching for fruit datasets...")
        
        # Look for common patterns
        patterns = [
            "*Archive*",
            "*archive*", 
            "*Fruit*",
            "*fruit*",
            "*Mango*",
            "*mango*",
            "*.zip",
            "*.rar",
            "*.7z"
        ]
        
        found_files = []
        for pattern in patterns:
            found_files.extend(list(self.base_path.glob(pattern)))
        
        logger.info(f"Found {len(found_files)} potential dataset files:")
        for file_path in found_files:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path.name}: {size_mb:.1f} MB")
            else:
                logger.info(f"  {file_path.name}: Directory")
        
        return found_files
    
    def identify_datasets(self, found_files):
        """Identify which files correspond to which datasets"""
        datasets = {}
        
        for file_path in found_files:
            name_lower = file_path.name.lower()
            
            if "archive" in name_lower and file_path.suffix in ['.zip', '.rar', '.7z']:
                datasets["moltean_fruits"] = {
                    "path": file_path,
                    "type": "archive",
                    "expected_size_gb": 4.13
                }
            elif "fruit" in name_lower and file_path.is_dir():
                datasets["shimul_fruits"] = {
                    "path": file_path,
                    "type": "directory",
                    "expected_size_mb": 314.3
                }
            elif "mango" in name_lower and (file_path.is_dir() or file_path.suffix in ['.zip', '.rar', '.7z']):
                datasets["mango_classify"] = {
                    "path": file_path,
                    "type": "archive" if file_path.suffix in ['.zip', '.rar', '.7z'] else "directory",
                    "expected_size_gb": 17.88
                }
        
        return datasets
    
    def extract_archive(self, archive_path, extract_to):
        """Extract archive file"""
        logger.info(f"📦 Extracting {archive_path.name}...")
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix == '.rar':
                # Note: Requires rarfile library for .rar files
                logger.warning("RAR extraction not supported. Please extract manually.")
                return False
            elif archive_path.suffix == '.7z':
                # Note: Requires py7zr library for .7z files
                logger.warning("7Z extraction not supported. Please extract manually.")
                return False
            
            logger.info(f"✅ Successfully extracted to {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error extracting {archive_path}: {str(e)}")
            return False
    
    def process_fruit_dataset(self, dataset_name, dataset_info):
        """Process a fruit dataset and convert to segmentation format"""
        logger.info(f"🔄 Processing {dataset_name}...")
        
        source_path = dataset_info["path"]
        
        # Extract if it's an archive
        if dataset_info["type"] == "archive":
            extract_path = self.processed_path / f"{dataset_name}_extracted"
            if not self.extract_archive(source_path, extract_path):
                return False
            source_path = extract_path
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(source_path.rglob(ext)))
        
        if not image_files:
            logger.warning(f"No image files found in {source_path}")
            return False
        
        logger.info(f"Found {len(image_files)} images in {dataset_name}")
        
        # Create processed directory
        processed_dataset_path = self.processed_path / dataset_name
        processed_dataset_path.mkdir(exist_ok=True)
        
        # Process images
        processed_count = 0
        class_mapping = {}
        class_counts = defaultdict(int)
        
        for img_path in image_files:
            try:
                # Determine class from directory structure
                relative_path = img_path.relative_to(source_path)
                class_name = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
                
                # Clean class name
                class_name = class_name.replace('_', ' ').replace('-', ' ').strip()
                
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)
                
                class_counts[class_name] += 1
                
                # Load and process image
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 512x512
                    img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # Save processed image
                    output_path = processed_dataset_path / f"{processed_count:06d}_{class_name.replace(' ', '_')}.png"
                    img_resized.save(output_path, 'PNG')
                    
                    processed_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Create segmentation masks
        self.create_segmentation_masks(processed_dataset_path, class_mapping)
        
        # Update metadata
        self.integration_metadata["datasets"][dataset_name] = {
            "original_images": len(image_files),
            "processed_images": processed_count,
            "classes": list(class_mapping.keys()),
            "class_count": len(class_mapping),
            "class_mapping": class_mapping,
            "class_counts": dict(class_counts),
            "processed_path": str(processed_dataset_path)
        }
        
        logger.info(f"✅ Processed {dataset_name}: {processed_count} images, {len(class_mapping)} classes")
        return True
    
    def create_segmentation_masks(self, dataset_path, class_mapping):
        """Create segmentation masks for classification datasets"""
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
                center_x, center_y = 256, 256
                width, height = 300, 300
                
                x1 = max(0, center_x - width // 2)
                y1 = max(0, center_y - height // 2)
                x2 = min(512, center_x + width // 2)
                y2 = min(512, center_y + height // 2)
                
                # Fill the rectangle with class ID
                class_name = img_path.stem.split('_')[-1].replace('_', ' ')
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
    
    def run_complete_processing(self):
        """Run the complete fruit dataset processing"""
        logger.info("🚀 Starting complete fruit dataset processing...")
        
        # Step 1: Find datasets
        found_files = self.find_fruit_datasets()
        
        # Step 2: Identify datasets
        datasets = self.identify_datasets(found_files)
        
        if not datasets:
            logger.error("No fruit datasets found. Please ensure the files are in the correct location.")
            return False
        
        logger.info(f"Identified {len(datasets)} fruit datasets:")
        for name, info in datasets.items():
            logger.info(f"  {name}: {info['path'].name}")
        
        # Step 3: Process each dataset
        for dataset_name, dataset_info in datasets.items():
            self.process_fruit_dataset(dataset_name, dataset_info)
        
        # Step 4: Integrate into main dataset
        if self.integration_metadata["datasets"]:
            self.integrate_into_main_dataset()
        
        # Step 5: Save integration metadata
        metadata_path = self.fruit_integration_path / "fruit_integration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.integration_metadata, f, indent=2)
        
        # Print summary
        self.print_integration_summary()
        
        return self.integration_metadata
    
    def print_integration_summary(self):
        """Print integration summary"""
        print("\n" + "="*80)
        print("🍎 FRUIT DATASET PROCESSING SUMMARY")
        print("="*80)
        
        total_images = sum(dataset["processed_images"] for dataset in self.integration_metadata["datasets"].values())
        total_classes = len(set().union(*[dataset["classes"] for dataset in self.integration_metadata["datasets"].values()]))
        
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"  Total Fruit Images: {total_images:,}")
        print(f"  Total Fruit Classes: {total_classes}")
        print(f"  Datasets Processed: {len(self.integration_metadata['datasets'])}")
        
        print(f"\n📁 DATASET BREAKDOWN:")
        for dataset_name, dataset_info in self.integration_metadata["datasets"].items():
            print(f"  {dataset_name}:")
            print(f"    Images: {dataset_info['processed_images']:,}")
            print(f"    Classes: {dataset_info['class_count']}")
            print(f"    Top Classes: {list(dataset_info['class_counts'].items())[:5]}")
        
        print(f"\n🏷️ ALL FRUIT CLASSES:")
        all_classes = set().union(*[dataset["classes"] for dataset in self.integration_metadata["datasets"].values()])
        print(f"  {', '.join(sorted(all_classes))}")
        
        print("\n" + "="*80)

def main():
    processor = ManualFruitDatasetProcessor()
    results = processor.run_complete_processing()
    
    print(f"\n🎉 Fruit processing complete!")
    print(f"📊 Total fruit images processed: {sum(dataset['processed_images'] for dataset in results['datasets'].values())}")

if __name__ == "__main__":
    main()
