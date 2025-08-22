"""
Visualization module for agricultural dataset exploration and quality assessment.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import get_image_files, ensure_directory


class Visualizer:
    """Class for visualizing agricultural datasets and processing results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color mapping for classes
        self.class_colors = {
            0: [0, 0, 0],      # background - black
            1: [0, 255, 0],    # crop - green
            2: [255, 0, 0],    # weed - red
            3: [255, 255, 0],  # partial_crop - yellow
            4: [139, 69, 19],  # soil - brown
            5: [128, 128, 128] # other - gray
        }
    
    def plot_dataset_overview(self, dataset_stats: Dict[str, Any], 
                            output_path: Optional[str] = None) -> None:
        """Create overview plots for all datasets."""
        self.logger.info("Creating dataset overview plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agricultural Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Dataset sizes (number of images)
        dataset_names = list(dataset_stats.keys())
        image_counts = [stats['total_images'] for stats in dataset_stats.values()]
        
        axes[0, 0].bar(dataset_names, image_counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Number of Images per Dataset')
        axes[0, 0].set_ylabel('Image Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Dataset sizes (storage)
        storage_sizes = [stats.get('statistics', {}).get('total_size_mb', 0) 
                        for stats in dataset_stats.values()]
        
        axes[0, 1].bar(dataset_names, storage_sizes, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Storage Size per Dataset')
        axes[0, 1].set_ylabel('Size (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Image format distribution
        format_data = {}
        for stats in dataset_stats.values():
            formats = stats.get('statistics', {}).get('formats', {})
            for fmt, count in formats.items():
                format_data[fmt] = format_data.get(fmt, 0) + count
        
        if format_data:
            axes[1, 0].pie(format_data.values(), labels=format_data.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Image Format Distribution')
        
        # 4. Annotation availability
        annotation_counts = [stats.get('total_annotations', 0) 
                           for stats in dataset_stats.values()]
        
        axes[1, 1].bar(dataset_names, annotation_counts, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Number of Annotations per Dataset')
        axes[1, 1].set_ylabel('Annotation Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            ensure_directory(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dataset overview saved to {output_path}")
        
        plt.show()
    
    def plot_image_samples(self, dataset_paths: Dict[str, str], 
                          num_samples: int = 3, output_path: Optional[str] = None) -> None:
        """Plot sample images from each dataset."""
        self.logger.info("Creating image sample plots...")
        
        num_datasets = len(dataset_paths)
        fig, axes = plt.subplots(num_datasets, num_samples, figsize=(15, 5 * num_datasets))
        fig.suptitle('Sample Images from Each Dataset', fontsize=16, fontweight='bold')
        
        if num_datasets == 1:
            axes = axes.reshape(1, -1)
        
        for i, (dataset_name, dataset_path) in enumerate(dataset_paths.items()):
            # Get sample images
            images_dir = os.path.join(dataset_path, 'Images') if 'Images' in os.listdir(dataset_path) else dataset_path
            if not os.path.exists(images_dir):
                images_dir = dataset_path
            
            image_files = get_image_files(images_dir, recursive=False)[:num_samples]
            
            for j, img_file in enumerate(image_files):
                try:
                    img_path = os.path.join(images_dir, img_file)
                    img = Image.open(img_path)
                    
                    # Resize for display if too large
                    if img.size[0] > 300 or img.size[1] > 300:
                        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f'{dataset_name}\n{os.path.basename(img_file)}')
                    axes[i, j].axis('off')
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_file}: {e}")
                    axes[i, j].text(0.5, 0.5, 'Image\nLoad Failed', 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            ensure_directory(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Image samples saved to {output_path}")
        
        plt.show()
    
    def plot_annotation_samples(self, dataset_paths: Dict[str, str], 
                              num_samples: int = 3, output_path: Optional[str] = None) -> None:
        """Plot sample annotations from each dataset."""
        self.logger.info("Creating annotation sample plots...")
        
        num_datasets = len(dataset_paths)
        fig, axes = plt.subplots(num_datasets, num_samples, figsize=(15, 5 * num_datasets))
        fig.suptitle('Sample Annotations from Each Dataset', fontsize=16, fontweight='bold')
        
        if num_datasets == 1:
            axes = axes.reshape(1, -1)
        
        for i, (dataset_name, dataset_path) in enumerate(dataset_paths.items()):
            # Look for annotation directories
            annotation_dirs = ['Labels', 'masks', 'annotations', 'semantics']
            annotation_dir = None
            
            for ann_dir in annotation_dirs:
                if os.path.exists(os.path.join(dataset_path, ann_dir)):
                    annotation_dir = os.path.join(dataset_path, ann_dir)
                    break
            
            if not annotation_dir:
                continue
            
            annotation_files = get_image_files(annotation_dir, recursive=False)[:num_samples]
            
            for j, ann_file in enumerate(annotation_files):
                try:
                    ann_path = os.path.join(annotation_dir, ann_file)
                    ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
                    
                    if ann is not None:
                        # Apply color mapping
                        colored_ann = self._apply_color_mapping(ann)
                        
                        # Resize for display if too large
                        if ann.shape[0] > 300 or ann.shape[1] > 300:
                            colored_ann = cv2.resize(colored_ann, (300, 300), interpolation=cv2.INTER_NEAREST)
                        
                        axes[i, j].imshow(colored_ann)
                        axes[i, j].set_title(f'{dataset_name}\n{os.path.basename(ann_file)}')
                        axes[i, j].axis('off')
                    else:
                        axes[i, j].text(0.5, 0.5, 'Annotation\nLoad Failed', 
                                       ha='center', va='center', transform=axes[i, j].transAxes)
                        axes[i, j].axis('off')
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load annotation {ann_file}: {e}")
                    axes[i, j].text(0.5, 0.5, 'Annotation\nLoad Failed', 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            ensure_directory(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Annotation samples saved to {output_path}")
        
        plt.show()
    
    def _apply_color_mapping(self, annotation: np.ndarray) -> np.ndarray:
        """Apply color mapping to annotation mask."""
        colored_ann = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
        
        for label_id, color in self.class_colors.items():
            mask = annotation == label_id
            colored_ann[mask] = color
        
        return colored_ann
    
    def plot_processing_results(self, processing_results: Dict[str, Any], 
                              output_path: Optional[str] = None) -> None:
        """Plot processing results and statistics."""
        self.logger.info("Creating processing results plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Processing Results', fontsize=16, fontweight='bold')
        
        # Extract data
        dataset_names = []
        original_images = []
        processed_images = []
        original_annotations = []
        processed_annotations = []
        
        for dataset_key, result in processing_results.items():
            if 'error' not in result:
                dataset_names.append(result['dataset_name'])
                original_images.append(result['total_images'])
                processed_images.append(result['processed_images'])
                original_annotations.append(result['total_annotations'])
                processed_annotations.append(result['processed_annotations'])
        
        # 1. Image processing success rate
        success_rates = [p/o*100 if o > 0 else 0 for p, o in zip(processed_images, original_images)]
        
        axes[0, 0].bar(dataset_names, success_rates, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Image Processing Success Rate')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=100, color='red', linestyle='--', alpha=0.7)
        
        # 2. Annotation processing success rate
        if any(original_annotations):
            ann_success_rates = [p/o*100 if o > 0 else 0 for p, o in zip(processed_annotations, original_annotations)]
            axes[0, 1].bar(dataset_names, ann_success_rates, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Annotation Processing Success Rate')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=100, color='red', linestyle='--', alpha=0.7)
        
        # 3. Original vs processed image counts
        x = np.arange(len(dataset_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, original_images, width, label='Original', color='lightcoral', alpha=0.7)
        axes[1, 0].bar(x + width/2, processed_images, width, label='Processed', color='lightblue', alpha=0.7)
        axes[1, 0].set_title('Original vs Processed Images')
        axes[1, 0].set_ylabel('Image Count')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(dataset_names, rotation=45)
        axes[1, 0].legend()
        
        # 4. Processing efficiency
        efficiency = [p/o if o > 0 else 0 for p, o in zip(processed_images, original_images)]
        axes[1, 1].bar(dataset_names, efficiency, color='gold', alpha=0.7)
        axes[1, 1].set_title('Processing Efficiency')
        axes[1, 1].set_ylabel('Efficiency Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if output_path:
            ensure_directory(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Processing results saved to {output_path}")
        
        plt.show()
    
    def plot_quality_metrics(self, quality_results: Dict[str, Any], 
                           output_path: Optional[str] = None) -> None:
        """Plot quality control metrics."""
        self.logger.info("Creating quality metrics plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Quality Metrics', fontsize=16, fontweight='bold')
        
        # Extract data
        splits = list(quality_results.keys())
        image_qualities = [quality_results[split]['average_image_quality'] for split in splits]
        annotation_qualities = [quality_results[split]['average_annotation_quality'] for split in splits]
        overall_qualities = [quality_results[split]['overall_quality'] for split in splits]
        total_images = [quality_results[split]['total_images'] for split in splits]
        
        # 1. Quality scores by split
        x = np.arange(len(splits))
        width = 0.25
        
        axes[0, 0].bar(x - width, image_qualities, width, label='Image Quality', color='lightblue', alpha=0.7)
        axes[0, 0].bar(x, annotation_qualities, width, label='Annotation Quality', color='lightgreen', alpha=0.7)
        axes[0, 0].bar(x + width, overall_qualities, width, label='Overall Quality', color='gold', alpha=0.7)
        axes[0, 0].set_title('Quality Scores by Split')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)
        
        # 2. Image count distribution
        axes[0, 1].pie(total_images, labels=splits, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Image Distribution Across Splits')
        
        # 3. Quality comparison
        quality_categories = ['Image', 'Annotation', 'Overall']
        avg_qualities = [
            np.mean(image_qualities),
            np.mean(annotation_qualities),
            np.mean(overall_qualities)
        ]
        
        axes[1, 0].bar(quality_categories, avg_qualities, color=['lightblue', 'lightgreen', 'gold'], alpha=0.7)
        axes[1, 0].set_title('Average Quality Across All Splits')
        axes[1, 0].set_ylabel('Average Quality Score')
        axes[1, 0].set_ylim(0, 1.1)
        
        # 4. Quality distribution histogram
        all_qualities = []
        for split in splits:
            all_qualities.extend([q['quality_score'] for q in quality_results[split]['image_quality_details']])
        
        if all_qualities:
            axes[1, 1].hist(all_qualities, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Distribution of Image Quality Scores')
            axes[1, 1].set_xlabel('Quality Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if output_path:
            ensure_directory(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Quality metrics saved to {output_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, dataset_stats: Dict[str, Any], 
                                   processing_results: Dict[str, Any],
                                   quality_results: Dict[str, Any]) -> None:
        """Create an interactive Plotly dashboard."""
        self.logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Dataset Sizes', 'Processing Success Rates', 
                          'Quality Metrics', 'Image Distribution', 'Storage Usage', 'Format Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Dataset sizes
        dataset_names = list(dataset_stats.keys())
        image_counts = [stats['total_images'] for stats in dataset_stats.values()]
        
        fig.add_trace(
            go.Bar(x=dataset_names, y=image_counts, name='Images', marker_color='skyblue'),
            row=1, col=1
        )
        
        # 2. Processing success rates
        success_rates = []
        for dataset_key in dataset_stats.keys():
            if dataset_key in processing_results and 'error' not in processing_results[dataset_key]:
                result = processing_results[dataset_key]
                rate = result['processed_images'] / result['total_images'] * 100 if result['total_images'] > 0 else 0
                success_rates.append(rate)
            else:
                success_rates.append(0)
        
        fig.add_trace(
            go.Bar(x=dataset_names, y=success_rates, name='Success Rate (%)', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # 3. Quality metrics
        if quality_results:
            splits = list(quality_results.keys())
            overall_qualities = [quality_results[split]['overall_quality'] for split in splits]
            
            fig.add_trace(
                go.Bar(x=splits, y=overall_qualities, name='Overall Quality', marker_color='gold'),
                row=2, col=1
            )
        
        # 4. Image distribution across splits
        if quality_results:
            total_images = [quality_results[split]['total_images'] for split in splits]
            
            fig.add_trace(
                go.Pie(labels=splits, values=total_images, name='Image Distribution'),
                row=2, col=2
            )
        
        # 5. Storage usage
        storage_sizes = [stats.get('statistics', {}).get('total_size_mb', 0) 
                        for stats in dataset_stats.values()]
        
        fig.add_trace(
            go.Bar(x=dataset_names, y=storage_sizes, name='Storage (MB)', marker_color='lightcoral'),
            row=3, col=1
        )
        
        # 6. Format distribution
        format_data = {}
        for stats in dataset_stats.values():
            formats = stats.get('statistics', {}).get('formats', {})
            for fmt, count in formats.items():
                format_data[fmt] = format_data.get(fmt, 0) + count
        
        if format_data:
            fig.add_trace(
                go.Pie(labels=list(format_data.keys()), values=list(format_data.values()), name='Formats'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Agricultural Dataset Combination Dashboard",
            showlegend=False,
            height=1200
        )
        
        # Show the dashboard
        fig.show()
        
        self.logger.info("Interactive dashboard created successfully!")
    
    def save_all_plots(self, output_dir: str, dataset_stats: Dict[str, Any],
                       processing_results: Dict[str, Any], quality_results: Dict[str, Any]) -> None:
        """Save all visualization plots to files."""
        ensure_directory(output_dir)
        
        # Save individual plots
        self.plot_dataset_overview(dataset_stats, os.path.join(output_dir, 'dataset_overview.png'))
        self.plot_processing_results(processing_results, os.path.join(output_dir, 'processing_results.png'))
        
        if quality_results:
            self.plot_quality_metrics(quality_results, os.path.join(output_dir, 'quality_metrics.png'))
        
        self.logger.info(f"All plots saved to {output_dir}")
