"""
Main dataset combiner module for agricultural datasets.
"""

import os
import json
import logging
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from pathlib import Path
import shutil

from dataset_loader import load_all_datasets, DatasetLoader
from preprocessing import Preprocessor, AnnotationMapper, QualityController
from utils import (
    ensure_directory, save_metadata, create_dataset_info,
    create_progress_tracker, backup_file
)


class DatasetCombiner:
    """Main class for combining multiple agricultural datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.preprocessor = Preprocessor(config)
        self.annotation_mapper = AnnotationMapper(config)
        self.quality_controller = QualityController(config)
        
        # Load all datasets
        self.dataset_loaders = load_all_datasets(config)
        
        # Output configuration
        self.output_dir = config['storage']['output_dir']
        self.intermediate_dir = config['storage']['intermediate_dir']
        self.backup_original = config['storage']['backup_original']
        
        # Data splitting configuration
        self.splits = config['splits']
        
        # Ensure output directories exist
        ensure_directory(self.output_dir)
        ensure_directory(self.intermediate_dir)
    
    def combine_datasets(self) -> Dict[str, Any]:
        """Main method to combine all datasets."""
        self.logger.info("Starting dataset combination process...")
        
        # Validate all datasets first
        self._validate_all_datasets()
        
        # Process each dataset
        processing_results = {}
        for dataset_key, loader in self.dataset_loaders.items():
            try:
                self.logger.info(f"Processing dataset: {dataset_key}")
                result = self._process_single_dataset(dataset_key, loader)
                processing_results[dataset_key] = result
            except Exception as e:
                self.logger.error(f"Failed to process dataset {dataset_key}: {e}")
                processing_results[dataset_key] = {'error': str(e)}
        
        # Combine processed datasets
        combined_result = self._combine_processed_datasets(processing_results)
        
        # Create final dataset structure
        final_result = self._create_final_dataset(combined_result)
        
        # Generate metadata and statistics
        metadata = self._generate_metadata(processing_results, final_result)
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        save_metadata(metadata, metadata_path)
        
        self.logger.info("Dataset combination completed successfully!")
        return {
            'processing_results': processing_results,
            'combined_result': combined_result,
            'final_result': final_result,
            'metadata': metadata
        }
    
    def _validate_all_datasets(self):
        """Validate all loaded datasets."""
        self.logger.info("Validating all datasets...")
        
        for dataset_key, loader in self.dataset_loaders.items():
            try:
                if not loader.validate_dataset():
                    self.logger.warning(f"Dataset {dataset_key} validation failed")
                else:
                    self.logger.info(f"Dataset {dataset_key} validation passed")
            except Exception as e:
                self.logger.error(f"Error validating dataset {dataset_key}: {e}")
    
    def _process_single_dataset(self, dataset_key: str, loader: DatasetLoader) -> Dict[str, Any]:
        """Process a single dataset."""
        dataset_config = self.config['datasets'][dataset_key]
        dataset_name = dataset_config['name']
        
        # Create intermediate output directory
        intermediate_output_dir = os.path.join(
            self.intermediate_dir, 
            f"processed_{dataset_key}"
        )
        
        # Process dataset
        processing_result = self.preprocessor.process_dataset(
            loader.get_images(),
            loader.get_annotations(),
            intermediate_output_dir,
            dataset_name
        )
        
        # Add dataset information
        processing_result.update({
            'dataset_key': dataset_key,
            'dataset_name': dataset_name,
            'source_path': dataset_config['source_path'],
            'original_statistics': loader.get_statistics(),
            'intermediate_output_dir': intermediate_output_dir
        })
        
        return processing_result
    
    def _combine_processed_datasets(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all processed datasets into unified structure."""
        self.logger.info("Combining processed datasets...")
        
        # Collect all processed data
        all_images = []
        all_annotations = []
        dataset_mapping = {}
        
        for dataset_key, result in processing_results.items():
            if 'error' in result:
                continue
            
            intermediate_dir = result['intermediate_output_dir']
            images_dir = os.path.join(intermediate_dir, 'images')
            annotations_dir = os.path.join(intermediate_dir, 'annotations')
            
            # Collect image files
            if os.path.exists(images_dir):
                for img_file in os.listdir(images_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(images_dir, img_file)
                        all_images.append(img_path)
                        dataset_mapping[img_file] = dataset_key
            
            # Collect annotation files
            if os.path.exists(annotations_dir):
                for ann_file in os.listdir(annotations_dir):
                    if ann_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ann_path = os.path.join(annotations_dir, ann_file)
                        all_annotations.append(ann_path)
        
        # Create combined directory structure
        combined_dir = os.path.join(self.intermediate_dir, 'combined')
        combined_images_dir = os.path.join(combined_dir, 'images')
        combined_annotations_dir = os.path.join(combined_dir, 'annotations')
        
        ensure_directory(combined_images_dir)
        ensure_directory(combined_annotations_dir)
        
        # Copy and rename files to avoid conflicts
        self.logger.info("Copying and organizing combined files...")
        progress_bar = create_progress_tracker(len(all_images), "Combining images")
        
        for i, img_path in enumerate(all_images):
            try:
                # Generate unique filename
                dataset_key = dataset_mapping[os.path.basename(img_path)]
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                new_filename = f"{dataset_key}_{base_name}_{i:06d}.png"
                
                # Copy image
                new_img_path = os.path.join(combined_images_dir, new_filename)
                shutil.copy2(img_path, new_img_path)
                
                # Copy corresponding annotation if exists
                old_ann_path = img_path.replace('/images/', '/annotations/')
                if os.path.exists(old_ann_path):
                    new_ann_filename = f"{dataset_key}_{base_name}_{i:06d}.png"
                    new_ann_path = os.path.join(combined_annotations_dir, new_ann_filename)
                    shutil.copy2(old_ann_path, new_ann_path)
                
                progress_bar.update(1)
                
            except Exception as e:
                self.logger.error(f"Error copying file {img_path}: {e}")
                progress_bar.update(1)
        
        progress_bar.close()
        
        return {
            'combined_directory': combined_dir,
            'total_images': len(all_images),
            'total_annotations': len(all_annotations),
            'dataset_mapping': dataset_mapping
        }
    
    def _create_final_dataset(self, combined_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create final dataset with train/val/test splits."""
        self.logger.info("Creating final dataset with splits...")
        
        combined_dir = combined_result['combined_directory']
        images_dir = os.path.join(combined_dir, 'images')
        annotations_dir = os.path.join(combined_dir, 'annotations')
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)  # Shuffle for random splitting
        
        # Calculate split indices
        total_images = len(image_files)
        train_end = int(total_images * self.splits['train'])
        val_end = train_end + int(total_images * self.splits['validation'])
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Create split directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, split_files in splits.items():
            split_dir = os.path.join(self.output_dir, split_name)
            split_images_dir = os.path.join(split_dir, 'images')
            split_annotations_dir = os.path.join(split_dir, 'annotations')
            
            ensure_directory(split_images_dir)
            ensure_directory(split_annotations_dir)
            
            # Copy files to split directories
            self.logger.info(f"Creating {split_name} split with {len(split_files)} images...")
            progress_bar = create_progress_tracker(len(split_files), f"Processing {split_name} split")
            
            for img_file in split_files:
                try:
                    # Copy image
                    src_img_path = os.path.join(images_dir, img_file)
                    dst_img_path = os.path.join(split_images_dir, img_file)
                    shutil.copy2(src_img_path, dst_img_path)
                    
                    # Copy annotation if exists
                    base_name = os.path.splitext(img_file)[0]
                    src_ann_path = os.path.join(annotations_dir, f"{base_name}.png")
                    if os.path.exists(src_ann_path):
                        dst_ann_path = os.path.join(split_annotations_dir, f"{base_name}.png")
                        shutil.copy2(src_ann_path, dst_ann_path)
                    
                    progress_bar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error copying file {img_file} to {split_name}: {e}")
                    progress_bar.update(1)
            
            progress_bar.close()
        
        return {
            'output_directory': self.output_dir,
            'splits': {
                'train': len(train_files),
                'val': len(val_files),
                'test': len(test_files)
            },
            'total_images': total_images
        }
    
    def _generate_metadata(self, processing_results: Dict[str, Any], 
                          final_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata for the combined dataset."""
        self.logger.info("Generating metadata...")
        
        # Dataset information
        dataset_info = {}
        for dataset_key, result in processing_results.items():
            if 'error' not in result:
                dataset_info[dataset_key] = {
                    'name': result['dataset_name'],
                    'source_path': result['source_path'],
                    'original_images': result['total_images'],
                    'processed_images': result['processed_images'],
                    'original_annotations': result['total_annotations'],
                    'processed_annotations': result['processed_annotations'],
                    'statistics': result['original_statistics']
                }
        
        # Combined dataset information
        combined_info = {
            'name': 'Combined Agricultural Dataset',
            'description': 'Combined dataset for Weakly Supervised Semantic Segmentation',
            'created_at': final_result.get('created_at', ''),
            'version': '1.0.0',
            'datasets': list(dataset_info.keys()),
            'total_images': final_result['total_images'],
            'splits': final_result['splits'],
            'output_directory': final_result['output_directory'],
            'unified_classes': self.annotation_mapper.get_unified_class_names(),
            'class_mapping': self.annotation_mapper.get_class_mapping()
        }
        
        # Processing information
        processing_info = {
            'config': self.config,
            'processing_results': processing_results,
            'quality_control': self._run_quality_control()
        }
        
        metadata = {
            'dataset_info': dataset_info,
            'combined_info': combined_info,
            'processing_info': processing_info
        }
        
        return metadata
    
    def _run_quality_control(self) -> Dict[str, Any]:
        """Run quality control on the final dataset."""
        self.logger.info("Running quality control...")
        
        quality_results = {}
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.output_dir, split)
            if os.path.exists(split_dir):
                quality_results[split] = self.quality_controller.validate_dataset_quality(split_dir)
        
        return quality_results
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all datasets."""
        stats = {}
        
        for dataset_key, loader in self.dataset_loaders.items():
            stats[dataset_key] = {
                'name': loader.dataset_name,
                'total_images': len(loader.get_images()),
                'total_annotations': len(loader.get_annotations()),
                'statistics': loader.get_statistics()
            }
        
        return stats
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate processing files."""
        if os.path.exists(self.intermediate_dir):
            shutil.rmtree(self.intermediate_dir)
            self.logger.info("Cleaned up intermediate files")
    
    def export_dataset_info(self, output_path: str):
        """Export dataset information to various formats."""
        ensure_directory(os.path.dirname(output_path))
        
        # Get statistics
        stats = self.get_dataset_statistics()
        
        # Export as JSON
        json_path = f"{output_path}.json"
        save_metadata(stats, json_path)
        
        # Export as CSV (basic format)
        csv_path = f"{output_path}.csv"
        self._export_to_csv(stats, csv_path)
        
        self.logger.info(f"Dataset information exported to {json_path} and {csv_path}")
    
    def _export_to_csv(self, stats: Dict[str, Any], csv_path: str):
        """Export statistics to CSV format."""
        import pandas as pd
        
        # Flatten statistics for CSV
        csv_data = []
        for dataset_key, dataset_stats in stats.items():
            row = {
                'dataset_key': dataset_key,
                'dataset_name': dataset_stats['name'],
                'total_images': dataset_stats['total_images'],
                'total_annotations': dataset_stats['total_annotations']
            }
            
            # Add statistics if available
            if 'statistics' in dataset_stats:
                stats_data = dataset_stats['statistics']
                row.update({
                    'avg_image_size': str(stats_data.get('avg_size', 'N/A')),
                    'total_size_mb': stats_data.get('total_size_mb', 0),
                    'formats': str(stats_data.get('formats', {}))
                })
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
