"""
Preprocessing module for agricultural dataset images and annotations.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from pathlib import Path

from utils import (
    resize_image, normalize_image, ensure_directory,
    create_progress_tracker, backup_file
)


class Preprocessor:
    """Base class for preprocessing agricultural dataset images and annotations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
        # Target image size
        self.target_size = tuple(config['output']['target_image_size'])
        self.output_format = config['output']['output_format']
        self.compression_quality = config['output']['compression_quality']
        
        # Processing options
        self.normalize = config['processing']['normalization']
        self.augment = config['processing']['augmentation']
        self.resize_method = config['processing']['resize_method']
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create data augmentation pipeline."""
        aug_config = self.config['processing']['augmentation_params']
        
        transforms = []
        
        if aug_config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if 'rotation_range' in aug_config:
            min_angle, max_angle = aug_config['rotation_range']
            transforms.append(A.Rotate(limit=(min_angle, max_angle), p=0.5))
        
        if 'brightness_range' in aug_config:
            min_brightness, max_brightness = aug_config['brightness_range']
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=(min_brightness, max_brightness),
                contrast_limit=(0.8, 1.2),
                p=0.5
            ))
        
        if 'saturation_range' in aug_config:
            min_sat, max_sat = aug_config['saturation_range']
            transforms.append(A.ColorJitter(
                saturation=(min_sat, max_sat),
                p=0.5
            ))
        
        if 'hue_range' in aug_config:
            min_hue, max_hue = aug_config['hue_range']
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=(min_hue, max_hue),
                p=0.5
            ))
        
        return A.Compose(transforms)
    
    def preprocess_image(self, image_path: str, output_path: str) -> bool:
        """Preprocess a single image."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = resize_image(image, self.target_size, self.resize_method)
            
            # Normalize if requested
            if self.normalize:
                image = normalize_image(image)
                # Convert back to uint8 for saving
                image = (image * 255).astype(np.uint8)
            
            # Save processed image
            ensure_directory(os.path.dirname(output_path))
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            return False
    
    def preprocess_annotation(self, annotation_path: str, output_path: str, 
                            annotation_type: str = 'semantic') -> bool:
        """Preprocess a single annotation."""
        try:
            # Load annotation
            if annotation_type == 'semantic':
                annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            else:
                annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
            
            if annotation is None:
                self.logger.error(f"Failed to load annotation: {annotation_path}")
                return False
            
            # Resize annotation
            annotation = resize_image(annotation, self.target_size, 'nearest')
            
            # Save processed annotation
            ensure_directory(os.path.dirname(output_path))
            cv2.imwrite(output_path, annotation)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preprocessing annotation {annotation_path}: {e}")
            return False
    
    def augment_image_and_annotation(self, image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to image and annotation pair."""
        if not self.augment:
            return image, annotation
        
        try:
            # Prepare data for albumentations
            data = {
                'image': image,
                'mask': annotation
            }
            
            # Apply augmentation
            augmented = self.augmentation_pipeline(**data)
            
            return augmented['image'], augmented['mask']
            
        except Exception as e:
            self.logger.warning(f"Augmentation failed: {e}, returning original")
            return image, annotation
    
    def process_dataset(self, image_paths: List[str], annotation_paths: List[str],
                       output_dir: str, dataset_name: str) -> Dict[str, Any]:
        """Process entire dataset."""
        self.logger.info(f"Processing dataset: {dataset_name}")
        
        # Create output directories
        images_output_dir = os.path.join(output_dir, 'images')
        annotations_output_dir = os.path.join(output_dir, 'annotations')
        ensure_directory(images_output_dir)
        ensure_directory(annotations_output_dir)
        
        # Statistics
        processed_images = 0
        processed_annotations = 0
        failed_images = 0
        failed_annotations = 0
        
        # Process images
        self.logger.info("Processing images...")
        progress_bar = create_progress_tracker(len(image_paths), f"Processing {dataset_name} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_image_path = os.path.join(
                    images_output_dir, 
                    f"{base_name}.{self.output_format}"
                )
                
                # Preprocess image
                if self.preprocess_image(image_path, output_image_path):
                    processed_images += 1
                else:
                    failed_images += 1
                
                progress_bar.update(1)
                
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
                failed_images += 1
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Process annotations if available
        if annotation_paths:
            self.logger.info("Processing annotations...")
            progress_bar = create_progress_tracker(len(annotation_paths), f"Processing {dataset_name} annotations")
            
            for i, annotation_path in enumerate(annotation_paths):
                try:
                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(annotation_path))[0]
                    output_annotation_path = os.path.join(
                        annotations_output_dir, 
                        f"{base_name}.{self.output_format}"
                    )
                    
                    # Determine annotation type
                    annotation_type = 'semantic'  # Default
                    if 'instance' in annotation_path.lower():
                        annotation_type = 'instance'
                    
                    # Preprocess annotation
                    if self.preprocess_annotation(annotation_path, output_annotation_path, annotation_type):
                        processed_annotations += 1
                    else:
                        failed_annotations += 1
                    
                    progress_bar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing annotation {annotation_path}: {e}")
                    failed_annotations += 1
                    progress_bar.update(1)
            
            progress_bar.close()
        
        # Summary
        summary = {
            'dataset_name': dataset_name,
            'total_images': len(image_paths),
            'total_annotations': len(annotation_paths),
            'processed_images': processed_images,
            'processed_annotations': processed_annotations,
            'failed_images': failed_images,
            'failed_annotations': failed_annotations,
            'output_directory': output_dir
        }
        
        self.logger.info(f"Dataset processing completed: {summary}")
        return summary


class AnnotationMapper:
    """Class for mapping annotations between different label schemes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load annotation mapping
        self.annotation_mapping = config['annotation_mapping']
        self.unified_classes = self.annotation_mapping['classes']
    
    def map_annotation(self, annotation: np.ndarray, source_dataset: str) -> np.ndarray:
        """Map annotation to unified label scheme."""
        if source_dataset not in self.annotation_mapping:
            self.logger.warning(f"No mapping found for dataset: {source_dataset}")
            return annotation
        
        # Get dataset-specific mapping
        dataset_mapping = self.annotation_mapping[source_dataset]
        
        # Create mapping array
        mapped_annotation = np.zeros_like(annotation)
        
        for old_label, new_label in dataset_mapping.items():
            if isinstance(old_label, int):
                mapped_annotation[annotation == old_label] = new_label
        
        return mapped_annotation
    
    def get_unified_class_names(self) -> List[str]:
        """Get list of unified class names."""
        return list(self.unified_classes.values())
    
    def get_class_mapping(self) -> Dict[str, int]:
        """Get class name to ID mapping."""
        return {name: idx for idx, name in self.unified_classes.items()}


class QualityController:
    """Class for quality control of processed datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_image_size = tuple(config['processing']['min_image_size'])
        self.max_image_size = tuple(config['processing']['max_image_size'])
        self.min_annotation_quality = config['processing']['min_annotation_quality']
    
    def check_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Check quality of a processed image."""
        try:
            with Image.open(image_path) as img:
                size = img.size
                mode = img.mode
                file_size = os.path.getsize(image_path)
                
                # Check size constraints
                size_ok = (self.min_image_size[0] <= size[0] <= self.max_image_size[0] and
                          self.min_image_size[1] <= size[1] <= self.max_image_size[1])
                
                # Check format
                format_ok = img.format.lower() in ['png', 'jpeg', 'jpg']
                
                quality_score = 1.0 if (size_ok and format_ok) else 0.5
                
                return {
                    'path': image_path,
                    'size': size,
                    'mode': mode,
                    'file_size': file_size,
                    'size_ok': size_ok,
                    'format_ok': format_ok,
                    'quality_score': quality_score
                }
                
        except Exception as e:
            return {
                'path': image_path,
                'error': str(e),
                'quality_score': 0.0
            }
    
    def check_annotation_quality(self, annotation_path: str) -> Dict[str, Any]:
        """Check quality of a processed annotation."""
        try:
            annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
            
            if annotation is None:
                return {
                    'path': annotation_path,
                    'error': 'Failed to load annotation',
                    'quality_score': 0.0
                }
            
            # Check for valid labels
            unique_labels = np.unique(annotation)
            valid_labels = unique_labels[unique_labels >= 0]
            
            # Calculate quality metrics
            label_coverage = len(valid_labels) / max(len(unique_labels), 1)
            quality_score = label_coverage if label_coverage >= self.min_annotation_quality else 0.0
            
            return {
                'path': annotation_path,
                'size': annotation.shape,
                'unique_labels': unique_labels.tolist(),
                'valid_labels': valid_labels.tolist(),
                'label_coverage': label_coverage,
                'quality_score': quality_score
            }
            
        except Exception as e:
            return {
                'path': annotation_path,
                'error': str(e),
                'quality_score': 0.0
            }
    
    def validate_dataset_quality(self, output_dir: str) -> Dict[str, Any]:
        """Validate quality of entire processed dataset."""
        self.logger.info(f"Validating dataset quality: {output_dir}")
        
        images_dir = os.path.join(output_dir, 'images')
        annotations_dir = os.path.join(output_dir, 'annotations')
        
        # Check images
        image_quality_scores = []
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(images_dir, img_file)
                    quality = self.check_image_quality(img_path)
                    image_quality_scores.append(quality)
        
        # Check annotations
        annotation_quality_scores = []
        if os.path.exists(annotations_dir):
            for ann_file in os.listdir(annotations_dir):
                if ann_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ann_path = os.path.join(annotations_dir, ann_file)
                    quality = self.check_annotation_quality(ann_path)
                    annotation_quality_scores.append(quality)
        
        # Calculate overall quality
        avg_image_quality = np.mean([q['quality_score'] for q in image_quality_scores]) if image_quality_scores else 0.0
        avg_annotation_quality = np.mean([q['quality_score'] for q in annotation_quality_scores]) if annotation_quality_scores else 0.0
        
        overall_quality = (avg_image_quality + avg_annotation_quality) / 2
        
        return {
            'output_directory': output_dir,
            'total_images': len(image_quality_scores),
            'total_annotations': len(annotation_quality_scores),
            'average_image_quality': avg_image_quality,
            'average_annotation_quality': avg_annotation_quality,
            'overall_quality': overall_quality,
            'image_quality_details': image_quality_scores,
            'annotation_quality_details': annotation_quality_scores
        }
