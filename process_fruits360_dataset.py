#!/usr/bin/env python3
"""
Process Fruits-360 Dataset
Processes the fruits-360 dataset and integrates it into the main agricultural dataset
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image
import numpy as np
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Fruits360Processor:
    def __init__(self, base_path="/Volumes/Rapid/Agriculture Dataset"):
        self.base_path = Path(base_path)
        self.fruits360_path = self.base_path / "fruits-360_original-size" / "fruits-360-original-size"
        self.processed_path = self.base_path / "Fruit_Integration" / "processed"
        self.integration_metadata = {
            "integration_date": datetime.now().isoformat(),
            "dataset_name": "fruits-360",
            "datasets": {},
            "total_images": 0,
            "total_classes": 0
        }
        
        # Create directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def analyze_dataset_structure(self):
        """Analyze the fruits-360 dataset structure"""
        logger.info("🔍 Analyzing fruits-360 dataset structure...")
        
        splits = ["Training", "Validation", "Test"]
        total_images = 0
        all_classes = set()
        
        for split in splits:
            split_path = self.fruits360_path / split
            if split_path.exists():
                # Count images in this split
                image_files = list(split_path.rglob("*.jpg")) + list(split_path.rglob("*.jpeg")) + list(split_path.rglob("*.png"))
                split_count = len(image_files)
                total_images += split_count
                
                # Get classes in this split
                classes = [d.name for d in split_path.iterdir() if d.is_dir()]
                all_classes.update(classes)
                
                logger.info(f"  {split}: {split_count:,} images, {len(classes)} classes")
            else:
                logger.warning(f"  {split}: Directory not found")
        
        logger.info(f"📊 Total: {total_images:,} images, {len(all_classes)} unique classes")
        return total_images, sorted(list(all_classes))
    
    def process_fruits360_dataset(self):
        """Process the fruits-360 dataset"""
        logger.info("🔄 Processing fruits-360 dataset...")
        
        # Analyze dataset first
        total_images, all_classes = self.analyze_dataset_structure()
        
        # Create processed directory
        processed_dataset_path = self.processed_path / "fruits360"
        processed_dataset_path.mkdir(exist_ok=True)
        
        # Process each split
        splits = ["Training", "Validation", "Test"]
        processed_count = 0
        class_mapping = {}
        class_counts = defaultdict(int)
        
        for split in splits:
            split_path = self.fruits360_path / split
            if not split_path.exists():
                continue
                
            logger.info(f"  Processing {split} split...")
            
            # Get all image files in this split
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(split_path.rglob(ext)))
            
            split_processed = 0
            
            for img_path in image_files:
                try:
                    # Determine class from directory structure
                    relative_path = img_path.relative_to(split_path)
                    class_name = relative_path.parts[0]
                    
                    # Clean class name
                    class_name = class_name.replace(' ', '_').replace('-', '_').strip()
                    
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
                        output_path = processed_dataset_path / f"{processed_count:06d}_{class_name}.png"
                        img_resized.save(output_path, 'PNG')
                        
                        processed_count += 1
                        split_processed += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {str(e)}")
                    continue
            
            logger.info(f"    Processed {split_processed:,} images from {split}")
        
        # Create segmentation masks
        self.create_segmentation_masks(processed_dataset_path, class_mapping)
        
        # Update metadata
        self.integration_metadata["datasets"]["fruits360"] = {
            "original_images": total_images,
            "processed_images": processed_count,
            "classes": list(class_mapping.keys()),
            "class_count": len(class_mapping),
            "class_mapping": class_mapping,
            "class_counts": dict(class_counts),
            "processed_path": str(processed_dataset_path)
        }
        
        self.integration_metadata["total_images"] = processed_count
        self.integration_metadata["total_classes"] = len(class_mapping)
        
        logger.info(f"✅ Processed fruits-360: {processed_count:,} images, {len(class_mapping)} classes")
        return processed_count
    
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
        """Integrate processed fruit dataset into main agricultural dataset"""
        logger.info("🔗 Integrating fruits-360 dataset into main agricultural dataset...")
        
        main_dataset_path = self.base_path / "Combined_datasets"
        if not main_dataset_path.exists():
            logger.error("Main dataset not found. Please run the main dataset combination first.")
            return False
        
        # Create fruit integration directory in main dataset
        fruit_main_path = main_dataset_path / "fruit_datasets"
        fruit_main_path.mkdir(exist_ok=True)
        
        # Copy processed dataset
        source_path = Path(self.integration_metadata["datasets"]["fruits360"]["processed_path"])
        target_path = fruit_main_path / "fruits360"
        
        if source_path.exists():
            # Copy dataset
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
            
            total_fruit_images = self.integration_metadata["total_images"]
            all_classes = set(self.integration_metadata["datasets"]["fruits360"]["classes"])
            
            logger.info(f"✅ Integrated fruits-360: {total_fruit_images:,} images")
            
            # Update main dataset metadata
            self.update_main_dataset_metadata(total_fruit_images, all_classes)
            
            return True
        else:
            logger.error(f"Processed dataset not found: {source_path}")
            return False
    
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
        """Run the complete fruits-360 processing"""
        logger.info("🚀 Starting complete fruits-360 processing...")
        
        # Step 1: Process the dataset
        processed_count = self.process_fruits360_dataset()
        
        # Step 2: Integrate into main dataset
        if processed_count > 0:
            self.integrate_into_main_dataset()
        
        # Step 3: Save integration metadata
        metadata_path = self.base_path / "Fruit_Integration" / "fruits360_integration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.integration_metadata, f, indent=2)
        
        # Print summary
        self.print_integration_summary()
        
        return self.integration_metadata
    
    def print_integration_summary(self):
        """Print integration summary"""
        print("\n" + "="*80)
        print("🍎 FRUITS-360 DATASET PROCESSING SUMMARY")
        print("="*80)
        
        total_images = self.integration_metadata["total_images"]
        total_classes = self.integration_metadata["total_classes"]
        
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"  Total Fruit Images: {total_images:,}")
        print(f"  Total Fruit Classes: {total_classes}")
        print(f"  Datasets Processed: 1")
        
        print(f"\n📁 DATASET BREAKDOWN:")
        for dataset_name, dataset_info in self.integration_metadata["datasets"].items():
            print(f"  {dataset_name}:")
            print(f"    Images: {dataset_info['processed_images']:,}")
            print(f"    Classes: {dataset_info['class_count']}")
            print(f"    Top Classes: {list(dataset_info['class_counts'].items())[:10]}")
        
        print(f"\n🏷️ SAMPLE FRUIT CLASSES:")
        all_classes = self.integration_metadata["datasets"]["fruits360"]["classes"]
        print(f"  {', '.join(sorted(all_classes)[:20])}...")
        print(f"  (and {len(all_classes) - 20} more classes)")
        
        print("\n" + "="*80)

def main():
    processor = Fruits360Processor()
    results = processor.run_complete_processing()
    
    print(f"\n🎉 Fruits-360 processing complete!")
    print(f"📊 Total fruit images processed: {results['total_images']:,}")

if __name__ == "__main__":
    main()
