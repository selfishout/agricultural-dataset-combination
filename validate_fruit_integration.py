#!/usr/bin/env python3
"""
Validation script for fruit datasets integration
Validates the quality and structure of integrated fruit datasets
"""

import os
import sys
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append('src')

from utils import setup_logging, load_config
from preprocessing import QualityController

class FruitIntegrationValidator:
    """Validates the fruit datasets integration"""
    
    def __init__(self, config_path="config/dataset_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging("INFO")
        
        # Integration directories
        self.integration_dir = "fruit_integration"
        self.processed_dir = os.path.join(self.integration_dir, "processed")
        self.final_dir = os.path.join(self.integration_dir, "final")
        self.validation_dir = os.path.join(self.integration_dir, "validation")
        
        # Create validation directory
        os.makedirs(self.validation_dir, exist_ok=True)
    
    def validate_processed_datasets(self):
        """Validate the processed fruit datasets"""
        print("🔍 Validating processed fruit datasets...")
        
        validation_results = {
            "processed_datasets": {},
            "total_images": 0,
            "total_classes": 0,
            "issues": [],
            "recommendations": []
        }
        
        if not os.path.exists(self.processed_dir):
            validation_results["issues"].append("Processed directory not found")
            return validation_results
        
        # Check each processed dataset
        for dataset_name in os.listdir(self.processed_dir):
            dataset_path = os.path.join(self.processed_dir, dataset_name)
            
            if not os.path.isdir(dataset_path):
                continue
            
            print(f"   📊 Validating {dataset_name}...")
            
            dataset_validation = self._validate_single_dataset(dataset_name, dataset_path)
            validation_results["processed_datasets"][dataset_name] = dataset_validation
            validation_results["total_images"] += dataset_validation.get("image_count", 0)
            validation_results["total_classes"] += dataset_validation.get("class_count", 0)
            
            # Collect issues
            if dataset_validation.get("issues"):
                validation_results["issues"].extend(dataset_validation["issues"])
        
        return validation_results
    
    def _validate_single_dataset(self, dataset_name, dataset_path):
        """Validate a single processed dataset"""
        validation = {
            "dataset_name": dataset_name,
            "path": dataset_path,
            "image_count": 0,
            "mask_count": 0,
            "class_count": 0,
            "classes": [],
            "image_sizes": [],
            "mask_sizes": [],
            "issues": [],
            "quality_score": 0.0
        }
        
        # Check directory structure
        images_dir = os.path.join(dataset_path, "images")
        masks_dir = os.path.join(dataset_path, "masks")
        metadata_file = os.path.join(dataset_path, "metadata.json")
        
        if not os.path.exists(images_dir):
            validation["issues"].append(f"Images directory not found: {images_dir}")
            return validation
        
        if not os.path.exists(masks_dir):
            validation["issues"].append(f"Masks directory not found: {masks_dir}")
            return validation
        
        if not os.path.exists(metadata_file):
            validation["issues"].append(f"Metadata file not found: {metadata_file}")
            return validation
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            validation.update(metadata)
        except Exception as e:
            validation["issues"].append(f"Error loading metadata: {e}")
        
        # Count actual files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
        
        validation["image_count"] = len(image_files)
        validation["mask_count"] = len(mask_files)
        
        # Validate image-mask pairs
        if validation["image_count"] != validation["mask_count"]:
            validation["issues"].append(f"Image-mask count mismatch: {validation['image_count']} images, {validation['mask_count']} masks")
        
        # Check image sizes
        for image_file in image_files[:10]:  # Sample first 10 images
            try:
                img_path = os.path.join(images_dir, image_file)
                with Image.open(img_path) as img:
                    validation["image_sizes"].append(img.size)
            except Exception as e:
                validation["issues"].append(f"Error reading image {image_file}: {e}")
        
        # Check mask sizes
        for mask_file in mask_files[:10]:  # Sample first 10 masks
            try:
                mask_path = os.path.join(masks_dir, mask_file)
                with Image.open(mask_path) as mask:
                    validation["mask_sizes"].append(mask.size)
            except Exception as e:
                validation["issues"].append(f"Error reading mask {mask_file}: {e}")
        
        # Calculate quality score
        quality_score = 1.0
        if validation["issues"]:
            quality_score -= len(validation["issues"]) * 0.1
        if validation["image_count"] == 0:
            quality_score = 0.0
        
        validation["quality_score"] = max(0.0, quality_score)
        
        return validation
    
    def validate_integration_output(self):
        """Validate the final integration output"""
        print("🔍 Validating integration output...")
        
        integration_output = os.path.join(self.final_dir, "integrated_agricultural_dataset")
        
        if not os.path.exists(integration_output):
            return {
                "status": "failed",
                "issues": ["Integration output directory not found"],
                "recommendations": ["Run the integration process first"]
            }
        
        validation = {
            "status": "success",
            "integration_path": integration_output,
            "splits": {},
            "total_images": 0,
            "total_masks": 0,
            "issues": [],
            "recommendations": []
        }
        
        # Check each split
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(integration_output, split)
            
            if not os.path.exists(split_path):
                validation["issues"].append(f"Split directory not found: {split}")
                continue
            
            split_validation = self._validate_split(split, split_path)
            validation["splits"][split] = split_validation
            validation["total_images"] += split_validation.get("image_count", 0)
            validation["total_masks"] += split_validation.get("mask_count", 0)
        
        return validation
    
    def _validate_split(self, split_name, split_path):
        """Validate a single split"""
        images_dir = os.path.join(split_path, "images")
        masks_dir = os.path.join(split_path, "masks")
        
        split_validation = {
            "split_name": split_name,
            "path": split_path,
            "image_count": 0,
            "mask_count": 0,
            "fruit_images": 0,
            "original_images": 0,
            "issues": []
        }
        
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
            split_validation["image_count"] = len(image_files)
            
            # Count fruit vs original images
            for img_file in image_files:
                if any(fruit in img_file.lower() for fruit in ['moltean', 'shimul', 'mango']):
                    split_validation["fruit_images"] += 1
                else:
                    split_validation["original_images"] += 1
        
        if os.path.exists(masks_dir):
            mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
            split_validation["mask_count"] = len(mask_files)
        
        if split_validation["image_count"] != split_validation["mask_count"]:
            split_validation["issues"].append(f"Image-mask count mismatch in {split_name}")
        
        return split_validation
    
    def create_validation_report(self, processed_validation, integration_validation):
        """Create a comprehensive validation report"""
        print("📊 Creating validation report...")
        
        report_path = os.path.join(self.validation_dir, "fruit_integration_validation_report.md")
        
        with open(report_path, 'w') as f:
            f.write(f"""# 🍎 Fruit Datasets Integration Validation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 Executive Summary

This report validates the fruit datasets integration process and ensures quality standards are met.

## 🔍 Processed Datasets Validation

### Overall Statistics
- **Total Processed Images**: {processed_validation.get('total_images', 0):,}
- **Total Classes**: {processed_validation.get('total_classes', 0)}
- **Datasets Processed**: {len(processed_validation.get('processed_datasets', {}))}

### Dataset Details
""")
            
            for dataset_name, dataset_validation in processed_validation.get('processed_datasets', {}).items():
                f.write(f"""
#### {dataset_validation.get('dataset_name', dataset_name)}
- **Images**: {dataset_validation.get('image_count', 0):,}
- **Masks**: {dataset_validation.get('mask_count', 0):,}
- **Classes**: {dataset_validation.get('class_count', 0)}
- **Quality Score**: {dataset_validation.get('quality_score', 0):.2f}
- **Issues**: {len(dataset_validation.get('issues', []))}
""")
                
                if dataset_validation.get('issues'):
                    f.write("   - Issues:\n")
                    for issue in dataset_validation['issues']:
                        f.write(f"     - {issue}\n")
            
            f.write(f"""
## 🔗 Integration Output Validation

### Overall Status
- **Status**: {integration_validation.get('status', 'unknown')}
- **Total Images**: {integration_validation.get('total_images', 0):,}
- **Total Masks**: {integration_validation.get('total_masks', 0):,}

### Split Distribution
""")
            
            for split_name, split_validation in integration_validation.get('splits', {}).items():
                f.write(f"""
#### {split_name.upper()} Split
- **Images**: {split_validation.get('image_count', 0):,}
- **Masks**: {split_validation.get('mask_count', 0):,}
- **Fruit Images**: {split_validation.get('fruit_images', 0):,}
- **Original Images**: {split_validation.get('original_images', 0):,}
""")
            
            f.write(f"""
## ⚠️ Issues Found

Total Issues: {len(processed_validation.get('issues', [])) + len(integration_validation.get('issues', []))}

### Processed Datasets Issues
""")
            
            for issue in processed_validation.get('issues', []):
                f.write(f"- {issue}\n")
            
            f.write(f"""
### Integration Issues
""")
            
            for issue in integration_validation.get('issues', []):
                f.write(f"- {issue}\n")
            
            f.write(f"""
## ✅ Recommendations

1. **Quality Assurance**: Ensure all images are properly processed
2. **Validation**: Run quality control checks regularly
3. **Monitoring**: Track processing statistics
4. **Backup**: Keep original datasets as backup

## 📊 Next Steps

1. Address any issues found in validation
2. Re-run integration if necessary
3. Test with sample segmentation project
4. Update documentation with final statistics

---
**Validation completed on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        
        print(f"✅ Validation report created: {report_path}")
        return report_path
    
    def create_validation_visualizations(self, processed_validation, integration_validation):
        """Create validation visualizations"""
        print("📊 Creating validation visualizations...")
        
        viz_dir = os.path.join(self.validation_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create validation overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fruit Integration Validation Overview', fontsize=16, fontweight='bold')
        
        # Dataset quality scores
        datasets = list(processed_validation.get('processed_datasets', {}).keys())
        quality_scores = [processed_validation['processed_datasets'][d].get('quality_score', 0) for d in datasets]
        
        axes[0, 0].bar(datasets, quality_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Dataset Quality Scores')
        axes[0, 0].set_ylabel('Quality Score (0-1)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # Image counts per dataset
        image_counts = [processed_validation['processed_datasets'][d].get('image_count', 0) for d in datasets]
        axes[0, 1].pie(image_counts, labels=datasets, autopct='%1.1f%%')
        axes[0, 1].set_title('Image Distribution by Dataset')
        
        # Split distribution
        splits = list(integration_validation.get('splits', {}).keys())
        split_images = [integration_validation['splits'][s].get('image_count', 0) for s in splits]
        
        axes[1, 0].bar(splits, split_images, color=['#FF9999', '#66B2FF', '#99FF99'])
        axes[1, 0].set_title('Split Distribution')
        axes[1, 0].set_ylabel('Number of Images')
        
        # Fruit vs Original images
        fruit_images = sum(integration_validation['splits'][s].get('fruit_images', 0) for s in splits)
        original_images = sum(integration_validation['splits'][s].get('original_images', 0) for s in splits)
        
        axes[1, 1].pie([fruit_images, original_images], 
                      labels=['Fruit Images', 'Original Images'], 
                      autopct='%1.1f%%',
                      colors=['#FFB366', '#66B2FF'])
        axes[1, 1].set_title('Fruit vs Original Images')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'validation_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Validation visualizations saved: {viz_dir}")
        return viz_dir
    
    def run_validation(self):
        """Run the complete validation process"""
        print("🔍 FRUIT INTEGRATION VALIDATION")
        print("=" * 50)
        
        # Validate processed datasets
        processed_validation = self.validate_processed_datasets()
        
        # Validate integration output
        integration_validation = self.validate_integration_output()
        
        # Create validation report
        report_path = self.create_validation_report(processed_validation, integration_validation)
        
        # Create visualizations
        viz_dir = self.create_validation_visualizations(processed_validation, integration_validation)
        
        # Print summary
        print(f"\n📊 VALIDATION SUMMARY")
        print(f"=" * 50)
        print(f"Processed datasets: {len(processed_validation.get('processed_datasets', {}))}")
        print(f"Total images processed: {processed_validation.get('total_images', 0):,}")
        print(f"Integration status: {integration_validation.get('status', 'unknown')}")
        print(f"Total issues found: {len(processed_validation.get('issues', [])) + len(integration_validation.get('issues', []))}")
        
        print(f"\n📁 Validation files:")
        print(f"   Report: {report_path}")
        print(f"   Visualizations: {viz_dir}")
        
        return processed_validation, integration_validation

def main():
    """Main validation function"""
    validator = FruitIntegrationValidator()
    processed_validation, integration_validation = validator.run_validation()
    
    # Determine overall status
    total_issues = len(processed_validation.get('issues', [])) + len(integration_validation.get('issues', []))
    
    if total_issues == 0:
        print("\n✅ VALIDATION PASSED: No issues found!")
    else:
        print(f"\n⚠️  VALIDATION WARNING: {total_issues} issues found. Check the report for details.")
    
    return total_issues == 0

if __name__ == "__main__":
    main()
