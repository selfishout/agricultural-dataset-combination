#!/usr/bin/env python3
"""
Comprehensive Fruit Datasets Integration
Integrates three fruit datasets into the existing agricultural dataset
"""

import os
import sys
import json
import shutil
from pathlib import Path
import zipfile
from PIL import Image, ImageDraw
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append('src')

from utils import setup_logging, load_config
from dataset_combiner import DatasetCombiner
from preprocessing import Preprocessor, AnnotationMapper, QualityController
from visualization import Visualizer

class FruitDatasetIntegrator:
    """Integrates fruit datasets into the agricultural dataset"""
    
    def __init__(self, config_path="config/dataset_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging("INFO")
        
        # Fruit dataset information
        self.fruit_datasets = {
            "moltean_fruits": {
                "name": "Fruits (Moltean)",
                "url": "https://www.kaggle.com/datasets/moltean/fruits",
                "kaggle_id": "moltean/fruits",
                "estimated_images": 1000,
                "estimated_classes": 10,
                "description": "Fruit images with classification labels"
            },
            "shimul_fruits": {
                "name": "Fruits Dataset (Shimul)",
                "url": "https://www.kaggle.com/datasets/shimulmbstu/fruitsdataset",
                "kaggle_id": "shimulmbstu/fruitsdataset", 
                "estimated_images": 1500,
                "estimated_classes": 15,
                "description": "Fruit classification dataset"
            },
            "mango_classify": {
                "name": "Mango Classify (12 Native Mango)",
                "url": "https://www.kaggle.com/datasets/researchersajid/mangoclassify-12-native-mango-dataset-from-bd",
                "kaggle_id": "researchersajid/mangoclassify-12-native-mango-dataset-from-bd",
                "estimated_images": 500,
                "estimated_classes": 12,
                "description": "12 native mango varieties from Bangladesh"
            }
        }
        
        # Integration settings
        self.integration_settings = {
            "target_size": (512, 512),
            "output_format": "PNG",
            "mask_creation_method": "full_image",  # For classification datasets
            "quality_threshold": 0.8,
            "duplicate_threshold": 0.95
        }
    
    def setup_directories(self):
        """Set up directories for fruit dataset integration"""
        print("📁 Setting up directories...")
        
        # Create main integration directory
        self.integration_dir = "fruit_integration"
        os.makedirs(self.integration_dir, exist_ok=True)
        
        # Create subdirectories
        self.download_dir = os.path.join(self.integration_dir, "downloads")
        self.processed_dir = os.path.join(self.integration_dir, "processed")
        self.final_dir = os.path.join(self.integration_dir, "final")
        
        for dir_path in [self.download_dir, self.processed_dir, self.final_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ Directories created: {self.integration_dir}")
    
    def download_instructions(self):
        """Provide instructions for manual download"""
        print("\n📥 FRUIT DATASETS DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("Since Kaggle API key is not configured, please download manually:")
        print()
        
        for dataset_key, dataset_info in self.fruit_datasets.items():
            print(f"🍎 {dataset_info['name']}")
            print(f"   URL: {dataset_info['url']}")
            print(f"   Estimated images: {dataset_info['estimated_images']}")
            print(f"   Estimated classes: {dataset_info['estimated_classes']}")
            print(f"   Save to: {os.path.join(self.download_dir, dataset_key)}")
            print()
        
        print("📋 Download Steps:")
        print("1. Visit each URL above")
        print("2. Click 'Download' button")
        print("3. Extract the downloaded ZIP files")
        print("4. Place extracted folders in the specified directories")
        print("5. Run this script again with --process flag")
        
        return True
    
    def process_fruit_dataset(self, dataset_key, dataset_path):
        """Process a single fruit dataset"""
        print(f"\n🔄 Processing {self.fruit_datasets[dataset_key]['name']}...")
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return None
        
        # Create output directory
        output_dir = os.path.join(self.processed_dir, dataset_key)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        images_dir = os.path.join(output_dir, "images")
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Process images
        processed_count = 0
        class_mapping = {}
        class_counter = Counter()
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_path = os.path.join(root, file)
                    
                    # Get class name from directory structure
                    class_name = os.path.basename(root)
                    if class_name.lower() in ['images', 'data', 'train', 'val', 'test', 'annotations']:
                        continue
                    
                    try:
                        # Load and process image
                        with Image.open(image_path) as img:
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Resize to target size
                            img_resized = img.resize(self.integration_settings["target_size"], Image.Resampling.LANCZOS)
                            
                            # Create new filename
                            new_filename = f"{class_name}_{file}"
                            new_filename = os.path.splitext(new_filename)[0] + '.png'
                            
                            # Save processed image
                            img_path = os.path.join(images_dir, new_filename)
                            img_resized.save(img_path, self.integration_settings["output_format"])
                            
                            # Create segmentation mask (full image for classification)
                            mask = Image.new('L', self.integration_settings["target_size"], 255)
                            mask_filename = new_filename.replace('.png', '_mask.png')
                            mask_path = os.path.join(masks_dir, mask_filename)
                            mask.save(mask_path)
                            
                            # Update counters
                            processed_count += 1
                            class_counter[class_name] += 1
                            
                            if class_name not in class_mapping:
                                class_mapping[class_name] = len(class_mapping)
                    
                    except Exception as e:
                        print(f"⚠️  Error processing {image_path}: {e}")
                        continue
        
        # Create metadata
        metadata = {
            "dataset_name": self.fruit_datasets[dataset_key]['name'],
            "processed_images": processed_count,
            "classes": list(class_mapping.keys()),
            "class_count": len(class_mapping),
            "class_mapping": class_mapping,
            "class_distribution": dict(class_counter),
            "target_size": self.integration_settings["target_size"],
            "processing_date": datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Processed {processed_count} images from {len(class_mapping)} classes")
        print(f"   Classes: {', '.join(class_mapping.keys())}")
        
        return {
            "output_dir": output_dir,
            "metadata": metadata,
            "images_dir": images_dir,
            "masks_dir": masks_dir
        }
    
    def integrate_with_existing_dataset(self, processed_datasets):
        """Integrate processed fruit datasets with existing agricultural dataset"""
        print("\n🔗 Integrating fruit datasets with existing agricultural dataset...")
        
        # Load existing dataset configuration
        existing_dataset_path = self.config['storage']['output_dir']
        
        if not os.path.exists(existing_dataset_path):
            print(f"❌ Existing dataset not found: {existing_dataset_path}")
            return False
        
        # Create integration directory
        integration_output = os.path.join(self.final_dir, "integrated_agricultural_dataset")
        os.makedirs(integration_output, exist_ok=True)
        
        # Copy existing dataset structure
        print("📋 Copying existing dataset structure...")
        for split in ['train', 'val', 'test']:
            src_split = os.path.join(existing_dataset_path, split)
            dst_split = os.path.join(integration_output, split)
            
            if os.path.exists(src_split):
                shutil.copytree(src_split, dst_split, dirs_exist_ok=True)
                print(f"   ✅ Copied {split} split")
        
        # Add fruit datasets to appropriate splits
        print("🍎 Adding fruit datasets to splits...")
        
        # Calculate split ratios
        total_fruit_images = sum(dataset['metadata']['processed_images'] for dataset in processed_datasets.values())
        train_count = int(total_fruit_images * 0.7)
        val_count = int(total_fruit_images * 0.2)
        test_count = total_fruit_images - train_count - val_count
        
        print(f"   Total fruit images: {total_fruit_images}")
        print(f"   Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        # Distribute fruit images across splits
        current_split = 'train'
        current_count = 0
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for dataset_key, dataset_info in processed_datasets.items():
            images_dir = dataset_info['images_dir']
            masks_dir = dataset_info['masks_dir']
            
            # Get all image files
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
            
            for image_file in image_files:
                # Determine target split
                if current_count >= train_count and current_split == 'train':
                    current_split = 'val'
                    current_count = 0
                elif current_count >= val_count and current_split == 'val':
                    current_split = 'test'
                    current_count = 0
                
                # Copy image and mask
                src_image = os.path.join(images_dir, image_file)
                src_mask = os.path.join(masks_dir, image_file.replace('.png', '_mask.png'))
                
                dst_split_dir = os.path.join(integration_output, current_split)
                dst_images_dir = os.path.join(dst_split_dir, "images")
                dst_masks_dir = os.path.join(dst_split_dir, "masks")
                
                os.makedirs(dst_images_dir, exist_ok=True)
                os.makedirs(dst_masks_dir, exist_ok=True)
                
                # Create unique filename
                unique_name = f"{dataset_key}_{image_file}"
                dst_image = os.path.join(dst_images_dir, unique_name)
                dst_mask = os.path.join(dst_masks_dir, unique_name.replace('.png', '_mask.png'))
                
                shutil.copy2(src_image, dst_image)
                if os.path.exists(src_mask):
                    shutil.copy2(src_mask, dst_mask)
                
                current_count += 1
                split_counts[current_split] += 1
        
        # Create updated metadata
        updated_metadata = {
            "integration_date": datetime.now().isoformat(),
            "original_datasets": list(processed_datasets.keys()),
            "fruit_datasets_added": {
                dataset_key: dataset_info['metadata'] 
                for dataset_key, dataset_info in processed_datasets.items()
            },
            "split_distribution": split_counts,
            "total_fruit_images": total_fruit_images
        }
        
        # Save updated metadata
        metadata_path = os.path.join(integration_output, "fruit_integration_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(updated_metadata, f, indent=2)
        
        print(f"✅ Integration complete!")
        print(f"   Output directory: {integration_output}")
        print(f"   Split distribution: {split_counts}")
        
        return integration_output
    
    def create_integration_report(self, processed_datasets, integration_output):
        """Create a comprehensive integration report"""
        print("\n📊 Creating integration report...")
        
        # Create visualizations
        viz_dir = os.path.join(self.integration_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Dataset overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fruit Datasets Integration Overview', fontsize=16, fontweight='bold')
        
        # Dataset sizes
        dataset_names = [self.fruit_datasets[key]['name'] for key in processed_datasets.keys()]
        dataset_sizes = [dataset['metadata']['processed_images'] for dataset in processed_datasets.values()]
        
        axes[0, 0].bar(dataset_names, dataset_sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Dataset Sizes')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Class distribution
        all_classes = []
        for dataset in processed_datasets.values():
            all_classes.extend(dataset['metadata']['classes'])
        
        class_counts = Counter(all_classes)
        axes[0, 1].pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Class Distribution')
        
        # Processing timeline
        processing_dates = [datetime.now().strftime('%Y-%m-%d')] * len(processed_datasets)
        axes[1, 0].barh(dataset_names, [1] * len(processed_datasets), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 0].set_title('Processing Status')
        axes[1, 0].set_xlabel('Completed')
        
        # Integration summary
        total_images = sum(dataset_sizes)
        total_classes = len(set(all_classes))
        
        summary_text = f"""
        Integration Summary:
        • Total Images: {total_images:,}
        • Total Classes: {total_classes}
        • Datasets Integrated: {len(processed_datasets)}
        • Output Directory: {integration_output}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Integration Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'fruit_integration_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed report
        report_path = os.path.join(self.integration_dir, "fruit_integration_report.md")
        with open(report_path, 'w') as f:
            f.write(f"""# Fruit Datasets Integration Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report documents the integration of three fruit datasets into the existing agricultural dataset for Weakly Supervised Semantic Segmentation (WSSS).

## Integrated Datasets

""")
            
            for dataset_key, dataset_info in processed_datasets.items():
                metadata = dataset_info['metadata']
                f.write(f"""### {metadata['dataset_name']}

- **Source**: {self.fruit_datasets[dataset_key]['url']}
- **Processed Images**: {metadata['processed_images']:,}
- **Classes**: {metadata['class_count']}
- **Class List**: {', '.join(metadata['classes'])}
- **Processing Date**: {metadata['processing_date']}

""")
            
            f.write(f"""## Integration Summary

- **Total Fruit Images Added**: {total_images:,}
- **Total Classes Added**: {total_classes}
- **Integration Method**: Classification to Segmentation Conversion
- **Output Directory**: {integration_output}

## Technical Details

- **Target Image Size**: {self.integration_settings['target_size']}
- **Output Format**: {self.integration_settings['output_format']}
- **Mask Creation Method**: {self.integration_settings['mask_creation_method']}

## Next Steps

1. Verify the integrated dataset quality
2. Update the main dataset configuration
3. Re-run the complete dataset combination process
4. Update visualizations and documentation

## Files Generated

- Integration Report: {report_path}
- Visualizations: {viz_dir}/
- Processed Datasets: {self.processed_dir}/
- Final Integration: {integration_output}/
""")
        
        print(f"✅ Integration report created: {report_path}")
        print(f"✅ Visualizations saved: {viz_dir}")
        
        return report_path
    
    def run_integration(self, process_datasets=False):
        """Run the complete integration process"""
        print("🍎 FRUIT DATASETS INTEGRATION")
        print("=" * 60)
        
        # Setup directories
        self.setup_directories()
        
        if not process_datasets:
            return self.download_instructions()
        
        # Check if datasets are downloaded
        processed_datasets = {}
        
        for dataset_key, dataset_info in self.fruit_datasets.items():
            dataset_path = os.path.join(self.download_dir, dataset_key)
            
            if os.path.exists(dataset_path):
                print(f"\n🔄 Processing {dataset_info['name']}...")
                result = self.process_fruit_dataset(dataset_key, dataset_path)
                if result:
                    processed_datasets[dataset_key] = result
            else:
                print(f"⚠️  Dataset not found: {dataset_path}")
                print(f"   Please download from: {dataset_info['url']}")
        
        if not processed_datasets:
            print("\n❌ No datasets found to process. Please download datasets first.")
            return False
        
        # Integrate with existing dataset
        integration_output = self.integrate_with_existing_dataset(processed_datasets)
        
        if integration_output:
            # Create integration report
            self.create_integration_report(processed_datasets, integration_output)
            
            print(f"\n🎉 FRUIT DATASETS INTEGRATION COMPLETE!")
            print(f"   Integrated datasets: {len(processed_datasets)}")
            print(f"   Output directory: {integration_output}")
            print(f"   Report: {os.path.join(self.integration_dir, 'fruit_integration_report.md')}")
            
            return True
        
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate fruit datasets into agricultural dataset')
    parser.add_argument('--process', action='store_true', 
                       help='Process downloaded datasets (requires datasets to be downloaded first)')
    
    args = parser.parse_args()
    
    integrator = FruitDatasetIntegrator()
    success = integrator.run_integration(process_datasets=args.process)
    
    if success:
        print("\n✅ Integration process completed successfully!")
    else:
        print("\n❌ Integration process failed or incomplete.")
    
    return success

if __name__ == "__main__":
    main()
