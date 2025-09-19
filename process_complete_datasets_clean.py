#!/usr/bin/env python3
"""
Complete Dataset Processing Script - Clean Version
Processes all datasets without augmentation to get accurate counts
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, base_path="/Volumes/Rapid/Agriculture Dataset"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "Combined_datasets_clean"
        self.metadata = {
            "processing_date": datetime.now().isoformat(),
            "datasets": {},
            "total_images": 0,
            "splits": {"train": 0, "val": 0, "test": 0}
        }
        
    def process_phenobench(self):
        """Process PhenoBench dataset - all directories"""
        logger.info("Processing PhenoBench dataset...")
        
        phenobench_path = self.base_path / "PhenoBench"
        total_images = 0
        
        # Process main PhenoBench directory
        main_path = phenobench_path / "PhenoBench"
        if main_path.exists():
            images = list(main_path.rglob("*.png")) + list(main_path.rglob("*.jpg")) + list(main_path.rglob("*.jpeg"))
            total_images += len(images)
            logger.info(f"PhenoBench main: {len(images)} images")
        
        # Process unlabelled_patches
        patches_path = phenobench_path / "unlabelled_patches"
        if patches_path.exists():
            images = list(patches_path.rglob("*.png")) + list(patches_path.rglob("*.jpg")) + list(patches_path.rglob("*.jpeg"))
            total_images += len(images)
            logger.info(f"PhenoBench patches: {len(images)} images")
        
        # Process unlabelled_patches_augmented
        patches_aug_path = phenobench_path / "unlabelled_patches_augmented"
        if patches_aug_path.exists():
            images = list(patches_aug_path.rglob("*.png")) + list(patches_aug_path.rglob("*.jpg")) + list(patches_aug_path.rglob("*.jpeg"))
            total_images += len(images)
            logger.info(f"PhenoBench patches augmented: {len(images)} images")
        
        self.metadata["datasets"]["phenobench"] = {
            "total_images": total_images,
            "directories": ["PhenoBench", "unlabelled_patches", "unlabelled_patches_augmented"]
        }
        
        logger.info(f"PhenoBench total: {total_images} images")
        return total_images
    
    def process_capsicum(self):
        """Process Capsicum dataset"""
        logger.info("Processing Capsicum dataset...")
        
        capsicum_path = self.base_path / "Synthetic and Empirical Capsicum Annuum Image Dataset_1_all" / "uuid_884958f5-b868-46e1-b3d8-a0b5d91b02c0"
        
        if not capsicum_path.exists():
            logger.warning("Capsicum dataset not found!")
            return 0
        
        # Count images in extracted directories
        total_images = 0
        
        # Look for extracted directories
        for item in capsicum_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                images = list(item.rglob("*.png")) + list(item.rglob("*.jpg")) + list(item.rglob("*.jpeg"))
                total_images += len(images)
                logger.info(f"Capsicum {item.name}: {len(images)} images")
        
        self.metadata["datasets"]["capsicum"] = {
            "total_images": total_images,
            "extracted_from": "data.zip"
        }
        
        logger.info(f"Capsicum total: {total_images} images")
        return total_images
    
    def process_weed_augmented(self):
        """Process Weed Augmented dataset"""
        logger.info("Processing Weed Augmented dataset...")
        
        weed_path = self.base_path / "weed_augmented"
        
        if not weed_path.exists():
            logger.warning("Weed Augmented dataset not found!")
            return 0
        
        images = list(weed_path.rglob("*.png")) + list(weed_path.rglob("*.jpg")) + list(weed_path.rglob("*.jpeg"))
        total_images = len(images)
        
        self.metadata["datasets"]["weed_augmented"] = {
            "total_images": total_images
        }
        
        logger.info(f"Weed Augmented total: {total_images} images")
        return total_images
    
    def process_vineyard(self):
        """Process Vineyard dataset"""
        logger.info("Processing Vineyard dataset...")
        
        vineyard_path = self.base_path / "Vineyard Canopy Images during Early Growth Stage"
        
        if not vineyard_path.exists():
            logger.warning("Vineyard dataset not found!")
            return 0
        
        images = list(vineyard_path.rglob("*.png")) + list(vineyard_path.rglob("*.jpg")) + list(vineyard_path.rglob("*.jpeg"))
        total_images = len(images)
        
        self.metadata["datasets"]["vineyard"] = {
            "total_images": total_images
        }
        
        logger.info(f"Vineyard total: {total_images} images")
        return total_images
    
    def process_all_datasets(self):
        """Process all datasets and create summary"""
        logger.info("Starting complete dataset processing...")
        
        # Process each dataset
        phenobench_count = self.process_phenobench()
        capsicum_count = self.process_capsicum()
        weed_count = self.process_weed_augmented()
        vineyard_count = self.process_vineyard()
        
        # Calculate totals
        total_images = phenobench_count + capsicum_count + weed_count + vineyard_count
        
        self.metadata["total_images"] = total_images
        self.metadata["splits"] = {
            "train": int(total_images * 0.7),
            "val": int(total_images * 0.2),
            "test": int(total_images * 0.1)
        }
        
        # Save metadata
        metadata_path = self.base_path / "complete_dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info("="*60)
        logger.info("COMPLETE DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"PhenoBench: {phenobench_count:,} images")
        logger.info(f"Capsicum: {capsicum_count:,} images")
        logger.info(f"Weed Augmented: {weed_count:,} images")
        logger.info(f"Vineyard: {vineyard_count:,} images")
        logger.info("-"*60)
        logger.info(f"TOTAL: {total_images:,} images")
        logger.info(f"Train Split: {self.metadata['splits']['train']:,} images (70%)")
        logger.info(f"Val Split: {self.metadata['splits']['val']:,} images (20%)")
        logger.info(f"Test Split: {self.metadata['splits']['test']:,} images (10%)")
        logger.info("="*60)
        
        return total_images

def main():
    processor = DatasetProcessor()
    total_images = processor.process_all_datasets()
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Total images found: {total_images:,}")
    print(f"📁 Metadata saved to: /Volumes/Rapid/Agriculture Dataset/complete_dataset_metadata.json")

if __name__ == "__main__":
    main()
