#!/usr/bin/env python3
"""
Main script for combining agricultural datasets.
This script orchestrates the entire dataset combination process.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, load_config, ensure_directory
from dataset_combiner import DatasetCombiner
from visualization import Visualizer


def main():
    """Main function for dataset combination."""
    parser = argparse.ArgumentParser(description='Combine agricultural datasets')
    parser.add_argument('--config', type=str, 
                       default='../config/dataset_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, 
                       default='../data/final/combined_dataset',
                       help='Output directory for combined dataset')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up intermediate files after processing')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = '../logs/combine_datasets.log'
    logger = setup_logging(args.log_level, log_file)
    
    logger.info("Starting agricultural dataset combination process...")
    start_time = time.time()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # Override output directory if specified
        if args.output:
            config['storage']['output_dir'] = args.output
        
        logger.info(f"Output directory: {config['storage']['output_dir']}")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create output directories
    ensure_directory(config['storage']['output_dir'])
    ensure_directory(config['storage']['intermediate_dir'])
    ensure_directory('../logs')
    
    try:
        # Initialize dataset combiner
        logger.info("Initializing dataset combiner...")
        combiner = DatasetCombiner(config)
        
        # Get initial dataset statistics
        logger.info("Collecting initial dataset statistics...")
        initial_stats = combiner.get_dataset_statistics()
        
        # Print initial statistics
        logger.info("\n" + "="*60)
        logger.info("INITIAL DATASET STATISTICS")
        logger.info("="*60)
        
        total_images = 0
        total_annotations = 0
        
        for dataset_key, stats in initial_stats.items():
            logger.info(f"\n{dataset_key}:")
            logger.info(f"  Name: {stats['name']}")
            logger.info(f"  Images: {stats['total_images']}")
            logger.info(f"  Annotations: {stats['total_annotations']}")
            
            total_images += stats['total_images']
            total_annotations += stats['total_annotations']
            
            # Print detailed statistics if available
            if 'statistics' in stats and stats['statistics']:
                stats_data = stats['statistics']
                logger.info(f"  Total Size: {stats_data.get('total_size_mb', 0):.2f} MB")
                if 'avg_size' in stats_data:
                    logger.info(f"  Average Image Size: {stats_data['avg_size']}")
                if 'formats' in stats_data:
                    logger.info(f"  Formats: {list(stats_data['formats'].keys())}")
        
        logger.info("\n" + "="*60)
        logger.info(f"TOTAL: {total_images} images, {total_annotations} annotations")
        logger.info("="*60)
        
        # Start combination process
        logger.info("\nStarting dataset combination process...")
        logger.info("This may take a while depending on dataset sizes...")
        
        # Combine datasets
        results = combiner.combine_datasets()
        
        # Print results summary
        logger.info("\n" + "="*60)
        logger.info("DATASET COMBINATION RESULTS")
        logger.info("="*60)
        
        # Processing results
        logger.info("\nProcessing Results:")
        for dataset_key, result in results['processing_results'].items():
            if 'error' not in result:
                logger.info(f"  {dataset_key}: {result['processed_images']}/{result['total_images']} images processed")
                if result['total_annotations'] > 0:
                    logger.info(f"           {result['processed_annotations']}/{result['total_annotations']} annotations processed")
            else:
                logger.error(f"  {dataset_key}: ERROR - {result['error']}")
        
        # Final results
        final_result = results['final_result']
        logger.info(f"\nFinal Dataset:")
        logger.info(f"  Output Directory: {final_result['output_directory']}")
        logger.info(f"  Total Images: {final_result['total_images']}")
        logger.info(f"  Splits:")
        for split_name, split_count in final_result['splits'].items():
            logger.info(f"    {split_name}: {split_count} images")
        
        # Quality control results
        if 'processing_info' in results and 'quality_control' in results['processing_info']:
            logger.info(f"\nQuality Control Results:")
            quality_results = results['processing_info']['quality_control']
            for split_name, quality_data in quality_results.items():
                logger.info(f"  {split_name}:")
                logger.info(f"    Image Quality: {quality_data['average_image_quality']:.3f}")
                logger.info(f"    Annotation Quality: {quality_data['average_annotation_quality']:.3f}")
                logger.info(f"    Overall Quality: {quality_data['overall_quality']:.3f}")
        
        # Export dataset information
        logger.info("\nExporting dataset information...")
        export_path = os.path.join(config['storage']['output_dir'], 'dataset_info')
        combiner.export_dataset_info(export_path)
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("\nGenerating visualizations...")
            try:
                visualizer = Visualizer(config)
                
                # Create visualization output directory
                viz_output_dir = os.path.join(config['storage']['output_dir'], 'visualizations')
                ensure_directory(viz_output_dir)
                
                # Generate and save plots
                visualizer.save_all_plots(
                    viz_output_dir,
                    initial_stats,
                    results['processing_results'],
                    results.get('processing_info', {}).get('quality_control', {})
                )
                
                # Create interactive dashboard
                logger.info("Creating interactive dashboard...")
                visualizer.create_interactive_dashboard(
                    initial_stats,
                    results['processing_results'],
                    results.get('processing_info', {}).get('quality_control', {})
                )
                
                logger.info(f"Visualizations saved to {viz_output_dir}")
                
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
        
        # Cleanup intermediate files if requested
        if args.cleanup:
            logger.info("\nCleaning up intermediate files...")
            combiner.cleanup_intermediate_files()
        
        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("\n" + "="*60)
        logger.info("DATASET COMBINATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total processing time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        logger.info(f"Output directory: {config['storage']['output_dir']}")
        logger.info(f"Combined dataset ready for Weakly Supervised Semantic Segmentation!")
        
        # Print next steps
        logger.info("\nNext steps:")
        logger.info("1. Review the combined dataset in the output directory")
        logger.info("2. Check the metadata.json file for detailed information")
        logger.info("3. Use the dataset for your WSSS training")
        logger.info("4. Consider running validation script to verify quality")
        
    except Exception as e:
        logger.error(f"Dataset combination failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
