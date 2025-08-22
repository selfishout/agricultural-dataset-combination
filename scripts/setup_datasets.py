#!/usr/bin/env python3
"""
Setup script for agricultural dataset combination project.
This script validates and prepares datasets for combination.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, load_config, ensure_directory
from dataset_loader import load_all_datasets


def main():
    """Main function for dataset setup."""
    parser = argparse.ArgumentParser(description='Setup agricultural datasets for combination')
    parser.add_argument('--source', type=str, 
                       default='/Volumes/Rapid/Agriculture Dataset',
                       help='Source directory containing datasets')
    parser.add_argument('--config', type=str, 
                       default='config/dataset_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, 
                       default='data/raw',
                       help='Output directory for raw data')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = 'logs/setup_datasets.log'
    logger = setup_logging(args.log_level, log_file)
    
    logger.info("Starting dataset setup process...")
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        logger.error(f"Source directory does not exist: {args.source}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create output directory
    ensure_directory(args.output)
    
    # Load and validate all datasets
    try:
        logger.info("Loading and validating datasets...")
        dataset_loaders = load_all_datasets(config)
        
        if not dataset_loaders:
            logger.error("No datasets could be loaded")
            sys.exit(1)
        
        # Validate each dataset
        validation_results = {}
        for dataset_key, loader in dataset_loaders.items():
            logger.info(f"Validating dataset: {dataset_key}")
            try:
                is_valid = loader.validate_dataset()
                validation_results[dataset_key] = {
                    'valid': is_valid,
                    'total_images': len(loader.get_images()),
                    'total_annotations': len(loader.get_annotations()),
                    'statistics': loader.get_statistics()
                }
                
                if is_valid:
                    logger.info(f"Dataset {dataset_key} validation passed")
                else:
                    logger.warning(f"Dataset {dataset_key} validation failed")
                    
            except Exception as e:
                logger.error(f"Error validating dataset {dataset_key}: {e}")
                validation_results[dataset_key] = {
                    'valid': False,
                    'error': str(e)
                }
        
        # Print validation summary
        logger.info("\n" + "="*50)
        logger.info("DATASET VALIDATION SUMMARY")
        logger.info("="*50)
        
        total_images = 0
        total_annotations = 0
        valid_datasets = 0
        
        for dataset_key, result in validation_results.items():
            if result['valid']:
                valid_datasets += 1
                total_images += result.get('total_images', 0)
                total_annotations += result.get('total_annotations', 0)
                
                logger.info(f"\n{dataset_key}:")
                logger.info(f"  Status: VALID")
                logger.info(f"  Images: {result.get('total_images', 0)}")
                logger.info(f"  Annotations: {result.get('total_annotations', 0)}")
                
                # Print statistics if available
                stats = result.get('statistics', {})
                if stats:
                    logger.info(f"  Total Size: {stats.get('total_size_mb', 0):.2f} MB")
                    logger.info(f"  Formats: {list(stats.get('formats', {}).keys())}")
            else:
                logger.warning(f"\n{dataset_key}:")
                logger.warning(f"  Status: INVALID")
                if 'error' in result:
                    logger.warning(f"  Error: {result['error']}")
        
        logger.info("\n" + "="*50)
        logger.info(f"SUMMARY: {valid_datasets}/{len(validation_results)} datasets valid")
        logger.info(f"Total Images: {total_images}")
        logger.info(f"Total Annotations: {total_annotations}")
        logger.info("="*50)
        
        # Save validation results
        import json
        validation_file = os.path.join(args.output, 'validation_results.json')
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        logger.info(f"Validation results saved to {validation_file}")
        
        # Check for special cases
        logger.info("\nChecking for special cases...")
        
        # Check Capsicum dataset for extraction
        capsicum_path = os.path.join(args.source, 'Synthetic and Empirical Capsicum Annuum Image Dataset_1_all')
        if os.path.exists(capsicum_path):
            data_zip = os.path.join(capsicum_path, 'data.zip')
            if os.path.exists(data_zip):
                zip_size = os.path.getsize(data_zip) / (1024**3)  # Convert to GB
                logger.info(f"Capsicum dataset found: data.zip ({zip_size:.2f} GB)")
                logger.info("  Note: This dataset needs extraction before processing")
                logger.info("  Consider extracting it manually or using unzip command")
        
        # Check PhenoBench structure
        phenobench_path = os.path.join(args.source, 'PhenoBench')
        if os.path.exists(phenobench_path):
            logger.info("PhenoBench dataset found:")
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(phenobench_path, 'PhenoBench', split)
                if os.path.exists(split_path):
                    images_dir = os.path.join(split_path, 'images')
                    if os.path.exists(images_dir):
                        num_images = len([f for f in os.listdir(images_dir) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        logger.info(f"  {split}: {num_images} images")
        
        logger.info("\nDataset setup completed successfully!")
        logger.info("You can now run the combination script:")
        logger.info("python scripts/combine_datasets.py --config config/dataset_config.yaml")
        
    except Exception as e:
        logger.error(f"Dataset setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
