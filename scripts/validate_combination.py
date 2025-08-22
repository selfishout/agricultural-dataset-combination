#!/usr/bin/env python3
"""
Validation script for the combined agricultural dataset.
This script checks the quality and integrity of the combined dataset.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, load_config, ensure_directory
from preprocessing import QualityController
from visualization import Visualizer


def main():
    """Main function for dataset validation."""
    parser = argparse.ArgumentParser(description='Validate combined agricultural dataset')
    parser.add_argument('--dataset', type=str, 
                       default='../data/final/combined_dataset',
                       help='Path to combined dataset')
    parser.add_argument('--config', type=str, 
                       default='../config/dataset_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, 
                       default='../data/validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate validation visualizations')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = '../logs/validate_combination.log'
    logger = setup_logging(args.log_level, log_file)
    
    logger.info("Starting dataset validation process...")
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Combined dataset not found: {args.dataset}")
        logger.error("Please run the combination script first:")
        logger.error("python scripts/combine_datasets.py --config config/dataset_config.yaml")
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
    
    try:
        # Initialize quality controller
        logger.info("Initializing quality controller...")
        quality_controller = QualityController(config)
        
        # Check dataset structure
        logger.info("Checking dataset structure...")
        required_dirs = ['train', 'val', 'test']
        missing_dirs = []
        
        for split in required_dirs:
            split_dir = os.path.join(args.dataset, split)
            if not os.path.exists(split_dir):
                missing_dirs.append(split_dir)
            else:
                # Check for required subdirectories
                images_dir = os.path.join(split_dir, 'images')
                annotations_dir = os.path.join(split_dir, 'annotations')
                
                if not os.path.exists(images_dir):
                    missing_dirs.append(images_dir)
                if not os.path.exists(annotations_dir):
                    missing_dirs.append(annotations_dir)
        
        if missing_dirs:
            logger.error("Missing required directories:")
            for missing_dir in missing_dirs:
                logger.error(f"  {missing_dir}")
            sys.exit(1)
        
        logger.info("Dataset structure validation passed")
        
        # Validate each split
        logger.info("\nValidating dataset splits...")
        validation_results = {}
        
        for split in required_dirs:
            split_dir = os.path.join(args.dataset, split)
            logger.info(f"\nValidating {split} split...")
            
            try:
                # Run quality validation
                quality_result = quality_controller.validate_dataset_quality(split_dir)
                validation_results[split] = quality_result
                
                # Print quality summary
                logger.info(f"  {split} split validation results:")
                logger.info(f"    Total Images: {quality_result['total_images']}")
                logger.info(f"    Total Annotations: {quality_result['total_annotations']}")
                logger.info(f"    Average Image Quality: {quality_result['average_image_quality']:.3f}")
                logger.info(f"    Average Annotation Quality: {quality_result['average_annotation_quality']:.3f}")
                logger.info(f"    Overall Quality: {quality_result['overall_quality']:.3f}")
                
            except Exception as e:
                logger.error(f"Error validating {split} split: {e}")
                validation_results[split] = {'error': str(e)}
        
        # Overall validation summary
        logger.info("\n" + "="*60)
        logger.info("OVERALL VALIDATION SUMMARY")
        logger.info("="*60)
        
        total_images = 0
        total_annotations = 0
        overall_quality_scores = []
        
        for split, result in validation_results.items():
            if 'error' not in result:
                total_images += result['total_images']
                total_annotations += result['total_annotations']
                overall_quality_scores.append(result['overall_quality'])
                
                logger.info(f"\n{split.upper()} SPLIT:")
                logger.info(f"  Images: {result['total_images']}")
                logger.info(f"  Annotations: {result['total_annotations']}")
                logger.info(f"  Overall Quality: {result['overall_quality']:.3f}")
            else:
                logger.error(f"\n{split.upper()} SPLIT: ERROR - {result['error']}")
        
        if overall_quality_scores:
            avg_overall_quality = sum(overall_quality_scores) / len(overall_quality_scores)
            logger.info(f"\nOVERALL STATISTICS:")
            logger.info(f"  Total Images: {total_images}")
            logger.info(f"  Total Annotations: {total_annotations}")
            logger.info(f"  Average Overall Quality: {avg_overall_quality:.3f}")
            
            # Quality assessment
            if avg_overall_quality >= 0.9:
                quality_level = "EXCELLENT"
            elif avg_overall_quality >= 0.8:
                quality_level = "GOOD"
            elif avg_overall_quality >= 0.7:
                quality_level = "ACCEPTABLE"
            elif avg_overall_quality >= 0.6:
                quality_level = "FAIR"
            else:
                quality_level = "POOR"
            
            logger.info(f"  Quality Assessment: {quality_level}")
        
        # Check for potential issues
        logger.info("\n" + "="*60)
        logger.info("QUALITY ISSUES ANALYSIS")
        logger.info("="*60)
        
        issues_found = False
        
        for split, result in validation_results.items():
            if 'error' in result:
                continue
            
            # Check image quality
            low_quality_images = [q for q in result['image_quality_details'] if q['quality_score'] < 0.8]
            if low_quality_images:
                logger.warning(f"  {split} split: {len(low_quality_images)} low quality images found")
                issues_found = True
            
            # Check annotation quality
            low_quality_annotations = [q for q in result['annotation_quality_details'] if q['quality_score'] < 0.8]
            if low_quality_annotations:
                logger.warning(f"  {split} split: {len(low_quality_annotations)} low quality annotations found")
                issues_found = True
        
        if not issues_found:
            logger.info("  No significant quality issues found")
        
        # Save validation results
        logger.info("\nSaving validation results...")
        import json
        validation_file = os.path.join(args.output, 'validation_results.json')
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        logger.info(f"Validation results saved to {validation_file}")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("\nGenerating validation visualizations...")
            try:
                visualizer = Visualizer(config)
                
                # Create visualization output directory
                viz_output_dir = os.path.join(args.output, 'visualizations')
                ensure_directory(viz_output_dir)
                
                # Generate quality metrics plots
                quality_plot_path = os.path.join(viz_output_dir, 'quality_metrics.png')
                visualizer.plot_quality_metrics(validation_results, quality_plot_path)
                
                logger.info(f"Validation visualizations saved to {viz_output_dir}")
                
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
        
        # Final assessment
        logger.info("\n" + "="*60)
        logger.info("VALIDATION COMPLETED")
        logger.info("="*60)
        
        if avg_overall_quality >= 0.8:
            logger.info("✅ Dataset validation PASSED")
            logger.info("The combined dataset is ready for use in Weakly Supervised Semantic Segmentation")
        elif avg_overall_quality >= 0.7:
            logger.info("⚠️  Dataset validation PASSED with WARNINGS")
            logger.info("The combined dataset is usable but may benefit from additional quality improvements")
        else:
            logger.warning("❌ Dataset validation FAILED")
            logger.warning("The combined dataset has significant quality issues and should be reviewed")
        
        logger.info(f"\nValidation results saved to: {args.output}")
        logger.info("Review the results and address any quality issues before using the dataset")
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
