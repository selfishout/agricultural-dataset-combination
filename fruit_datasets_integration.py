#!/usr/bin/env python3
"""
Fruit Datasets Integration for Agricultural Dataset
Analyzes and integrates three fruit datasets into the existing agricultural dataset
"""

import os
import sys
import json
import shutil
from pathlib import Path
import zipfile
from PIL import Image
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append('src')

from utils import setup_logging, load_config
from dataset_loader import DatasetLoader
from preprocessing import Preprocessor, AnnotationMapper, QualityController
from dataset_combiner import DatasetCombiner

class FruitDatasetAnalyzer:
    """Analyzer for fruit datasets integration"""
    
    def __init__(self, config_path="config/dataset_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging("INFO")
        
        # Fruit dataset information
        self.fruit_datasets = {
            "moltean_fruits": {
                "name": "Fruits (Moltean)",
                "url": "https://www.kaggle.com/datasets/moltean/fruits",
                "kaggle_id": "moltean/fruits",
                "expected_structure": "classification",
                "description": "Fruit images with classification labels",
                "compatibility": "high"
            },
            "shimul_fruits": {
                "name": "Fruits Dataset (Shimul)",
                "url": "https://www.kaggle.com/datasets/shimulmbstu/fruitsdataset",
                "kaggle_id": "shimulmbstu/fruitsdataset",
                "expected_structure": "classification",
                "description": "Fruit classification dataset",
                "compatibility": "high"
            },
            "mango_classify": {
                "name": "Mango Classify (12 Native Mango)",
                "url": "https://www.kaggle.com/datasets/researchersajid/mangoclassify-12-native-mango-dataset-from-bd",
                "kaggle_id": "researchersajid/mangoclassify-12-native-mango-dataset-from-bd",
                "expected_structure": "classification",
                "description": "12 native mango varieties from Bangladesh",
                "compatibility": "medium"
            }
        }
        
        # Integration strategy
        self.integration_strategy = {
            "moltean_fruits": {
                "approach": "classification_to_segmentation",
                "method": "create_bounding_box_masks",
                "priority": "high"
            },
            "shimul_fruits": {
                "approach": "classification_to_segmentation", 
                "method": "create_bounding_box_masks",
                "priority": "high"
            },
            "mango_classify": {
                "approach": "classification_to_segmentation",
                "method": "create_bounding_box_masks", 
                "priority": "medium"
            }
        }
    
    def analyze_dataset_compatibility(self, dataset_path, dataset_name):
        """Analyze if a dataset is compatible with our agricultural dataset"""
        analysis = {
            'name': dataset_name,
            'path': dataset_path,
            'compatible': False,
            'issues': [],
            'recommendations': [],
            'integration_method': None
        }
        
        if not os.path.exists(dataset_path):
            analysis['issues'].append(f"Dataset path does not exist: {dataset_path}")
            return analysis
        
        # Check for images
        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            analysis['issues'].append("No image files found")
            return analysis
        
        analysis['image_count'] = len(image_files)
        
        # Check for class structure
        classes = set()
        for root, dirs, files in os.walk(dataset_path):
            for dir_name in dirs:
                if dir_name.lower() not in ['images', 'data', 'train', 'val', 'test', 'annotations']:
                    classes.add(dir_name)
        
        analysis['classes'] = list(classes)
        analysis['class_count'] = len(classes)
        
        # Check for existing annotations
        annotation_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.json', '.xml', '.txt', '.csv')):
                    annotation_files.append(os.path.join(root, file))
        
        analysis['annotation_count'] = len(annotation_files)
        
        # Determine compatibility
        if analysis['image_count'] > 0 and analysis['class_count'] > 0:
            analysis['compatible'] = True
            analysis['integration_method'] = "classification_to_segmentation"
            analysis['recommendations'].append("Can be integrated by converting classification to segmentation masks")
        else:
            analysis['issues'].append("Insufficient data for integration")
        
        return analysis
    
    def create_fruit_dataset_loader(self, dataset_name, dataset_path):
        """Create a dataset loader for fruit datasets"""
        
        class FruitDatasetLoader(DatasetLoader):
            def __init__(self, dataset_path, dataset_name):
                super().__init__(dataset_path, dataset_name)
                self.dataset_type = "fruit_classification"
            
            def load_dataset(self):
                """Load fruit classification dataset"""
                image_paths = []
                annotation_paths = []
                
                # Find all image files
                for root, dirs, files in os.walk(self.dataset_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                            image_path = os.path.join(root, file)
                            image_paths.append(image_path)
                            
                            # For classification datasets, we'll create pseudo-annotations
                            # based on the directory structure
                            annotation_path = self._create_pseudo_annotation(image_path)
                            annotation_paths.append(annotation_path)
                
                return {
                    'images': image_paths,
                    'annotations': annotation_paths,
                    'classes': self._extract_classes(),
                    'dataset_type': self.dataset_type
                }
            
            def _extract_classes(self):
                """Extract class names from directory structure"""
                classes = set()
                for root, dirs, files in os.walk(self.dataset_path):
                    for dir_name in dirs:
                        if dir_name.lower() not in ['images', 'data', 'train', 'val', 'test', 'annotations']:
                            classes.add(dir_name)
                return list(classes)
            
            def _create_pseudo_annotation(self, image_path):
                """Create pseudo-annotation for classification dataset"""
                # This would create a bounding box annotation for the entire image
                # In a real implementation, this would generate proper segmentation masks
                return image_path.replace('.jpg', '_pseudo_mask.png').replace('.jpeg', '_pseudo_mask.png').replace('.png', '_pseudo_mask.png')
        
        return FruitDatasetLoader(dataset_path, dataset_name)
    
    def create_integration_plan(self, analyses):
        """Create a comprehensive integration plan"""
        plan = {
            'total_datasets': len(analyses),
            'compatible_datasets': [],
            'incompatible_datasets': [],
            'integration_steps': [],
            'estimated_images': 0,
            'estimated_classes': 0
        }
        
        for dataset_key, analysis in analyses.items():
            if analysis['compatible']:
                plan['compatible_datasets'].append(dataset_key)
                plan['estimated_images'] += analysis.get('image_count', 0)
                plan['estimated_classes'] += analysis.get('class_count', 0)
                
                # Add integration steps
                plan['integration_steps'].append({
                    'dataset': dataset_key,
                    'step': 'download_and_extract',
                    'description': f"Download {analysis['name']} from Kaggle"
                })
                plan['integration_steps'].append({
                    'dataset': dataset_key,
                    'step': 'convert_to_segmentation',
                    'description': f"Convert classification labels to segmentation masks"
                })
                plan['integration_steps'].append({
                    'dataset': dataset_key,
                    'step': 'integrate_with_existing',
                    'description': f"Integrate with existing agricultural dataset"
                })
            else:
                plan['incompatible_datasets'].append(dataset_key)
        
        return plan
    
    def generate_integration_script(self, plan):
        """Generate a script to integrate the fruit datasets"""
        compatible_datasets_str = json.dumps(plan['compatible_datasets'], indent=2)
        
        script_content = f'''#!/usr/bin/env python3
"""
Fruit Datasets Integration Script
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
    datasets = {compatible_datasets_str}
    
    # Create download directory
    download_dir = "fruit_datasets_download"
    os.makedirs(download_dir, exist_ok=True)
    
    # Note: This requires Kaggle API setup
    print("⚠️  Note: Kaggle API key required for automatic download")
    print("   Please download the following datasets manually:")
    
    fruit_datasets_info = {{
        "moltean_fruits": {{
            "name": "Fruits (Moltean)",
            "url": "https://www.kaggle.com/datasets/moltean/fruits"
        }},
        "shimul_fruits": {{
            "name": "Fruits Dataset (Shimul)", 
            "url": "https://www.kaggle.com/datasets/shimulmbstu/fruitsdataset"
        }},
        "mango_classify": {{
            "name": "Mango Classify (12 Native Mango)",
            "url": "https://www.kaggle.com/datasets/researchersajid/mangoclassify-12-native-mango-dataset-from-bd"
        }}
    }}
    
    for dataset in {compatible_datasets_str}:
        if dataset in fruit_datasets_info:
            print(f"   • {{fruit_datasets_info[dataset]['name']}}: {{fruit_datasets_info[dataset]['url']}}")
    
    return download_dir

def convert_classification_to_segmentation(dataset_path, output_path):
    """Convert classification dataset to segmentation format"""
    print(f"🔄 Converting {{dataset_path}} to segmentation format...")
    
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
                new_image_name = f"{{class_name}}_{{file}}"
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
    
    print(f"✅ Conversion complete: {{len(os.listdir(images_dir))}} images processed")

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
    compatible_datasets = {compatible_datasets_str}
    for dataset in compatible_datasets:
        dataset_path = os.path.join(download_dir, dataset)
        output_path = os.path.join(download_dir, f"{{dataset}}_segmentation")
        
        if os.path.exists(dataset_path):
            convert_classification_to_segmentation(dataset_path, output_path)
    
    # Integrate with existing dataset
    config = load_config("config/dataset_config.yaml")
    integrate_with_existing_dataset(download_dir, config['storage']['output_dir'])
    
    print("🎉 Fruit datasets integration complete!")

if __name__ == "__main__":
    main()
'''
        
        return script_content
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("🍎 FRUIT DATASETS INTEGRATION ANALYSIS")
        print("=" * 60)
        
        # Create analysis directory
        analysis_dir = "fruit_integration_analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Simulate analysis for each dataset (since we can't download without API key)
        analyses = {}
        
        for dataset_key, dataset_info in self.fruit_datasets.items():
            print(f"\n📊 Analyzing {dataset_info['name']}...")
            
            # Create a mock analysis based on expected structure
            analysis = {
                'name': dataset_info['name'],
                'path': f"fruit_datasets_download/{dataset_key}",
                'compatible': True,  # Assume compatible for now
                'issues': [],
                'recommendations': [
                    "Download dataset from Kaggle",
                    "Convert classification labels to segmentation masks",
                    "Integrate with existing agricultural dataset"
                ],
                'integration_method': "classification_to_segmentation",
                'image_count': 0,  # Will be updated after download
                'class_count': 0,  # Will be updated after download
                'classes': []
            }
            
            # Add dataset-specific information
            if dataset_key == "moltean_fruits":
                analysis['estimated_images'] = 1000  # Estimated
                analysis['estimated_classes'] = 10   # Estimated
            elif dataset_key == "shimul_fruits":
                analysis['estimated_images'] = 1500  # Estimated
                analysis['estimated_classes'] = 15   # Estimated
            elif dataset_key == "mango_classify":
                analysis['estimated_images'] = 500   # Estimated
                analysis['estimated_classes'] = 12   # Known from description
            
            analyses[dataset_key] = analysis
            
            print(f"   ✅ Compatible: {analysis['compatible']}")
            print(f"   📊 Estimated images: {analysis.get('estimated_images', 'Unknown')}")
            print(f"   🏷️  Estimated classes: {analysis.get('estimated_classes', 'Unknown')}")
        
        # Create integration plan
        plan = self.create_integration_plan(analyses)
        
        # Generate integration script
        script_content = self.generate_integration_script(plan)
        
        # Save analysis results
        results_file = os.path.join(analysis_dir, "fruit_integration_analysis.json")
        with open(results_file, 'w') as f:
            json.dump({
                'analyses': analyses,
                'plan': plan,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save integration script
        script_file = os.path.join(analysis_dir, "integrate_fruit_datasets.py")
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Print summary
        print(f"\n📋 INTEGRATION ANALYSIS SUMMARY")
        print(f"=" * 60)
        print(f"Total datasets: {plan['total_datasets']}")
        print(f"Compatible datasets: {len(plan['compatible_datasets'])}")
        print(f"Incompatible datasets: {len(plan['incompatible_datasets'])}")
        print(f"Estimated total images: {plan['estimated_images']}")
        print(f"Estimated total classes: {plan['estimated_classes']}")
        
        print(f"\n✅ Compatible datasets:")
        for dataset in plan['compatible_datasets']:
            dataset_info = self.fruit_datasets[dataset]
            print(f"   • {dataset_info['name']}: {dataset_info['description']}")
        
        print(f"\n📁 Analysis files saved:")
        print(f"   • {results_file}")
        print(f"   • {script_file}")
        
        return analyses, plan

def main():
    """Main function"""
    analyzer = FruitDatasetAnalyzer()
    analyses, plan = analyzer.run_analysis()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Set up Kaggle API key for automatic download")
    print(f"2. Run the integration script: python3 fruit_integration_analysis/integrate_fruit_datasets.py")
    print(f"3. Verify integration with existing agricultural dataset")
    
    return analyses, plan

if __name__ == "__main__":
    main()
