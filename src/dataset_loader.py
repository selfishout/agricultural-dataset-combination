"""
Dataset loader module for handling different agricultural dataset formats.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from utils import (
    get_image_files, validate_image_file, get_image_info,
    calculate_dataset_statistics, ensure_directory
)


class DatasetLoader:
    """Base class for loading different types of agricultural datasets."""
    
    def __init__(self, dataset_path: str, dataset_name: str):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(f"{__name__}.{dataset_name}")
        
        # Dataset information
        self.image_paths = []
        self.annotation_paths = []
        self.dataset_info = {}
        self.statistics = {}
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset structure and files."""
        raise NotImplementedError("Subclasses must implement _load_dataset")
    
    def get_images(self) -> List[str]:
        """Get list of image file paths."""
        return self.image_paths
    
    def get_annotations(self) -> List[str]:
        """Get list of annotation file paths."""
        return self.annotation_paths
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return self.dataset_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.statistics
    
    def validate_dataset(self) -> bool:
        """Validate dataset integrity."""
        raise NotImplementedError("Subclasses must implement validate_dataset")


class PhenoBenchLoader(DatasetLoader):
    """Loader for PhenoBench dataset."""
    
    def _load_dataset(self):
        """Load PhenoBench dataset structure."""
        self.logger.info(f"Loading PhenoBench dataset from {self.dataset_path}")
        
        # PhenoBench has train/val/test splits
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_path = os.path.join(self.dataset_path, 'PhenoBench', split)
            if os.path.exists(split_path):
                self._load_split(split, split_path)
        
        # Calculate statistics
        self.statistics = calculate_dataset_statistics(self.image_paths)
        self.logger.info(f"Loaded {len(self.image_paths)} images from PhenoBench")
    
    def _load_split(self, split_name: str, split_path: str):
        """Load a specific split of the dataset."""
        images_dir = os.path.join(split_path, 'images')
        if os.path.exists(images_dir):
            images = get_image_files(images_dir, recursive=False)
            for img_path in images:
                self.image_paths.append(img_path)
                
                # Store split information
                img_name = os.path.basename(img_path)
                self.dataset_info[img_name] = {
                    'split': split_name,
                    'source': 'phenobench',
                    'has_annotations': split_name in ['train', 'val']
                }
    
    def validate_dataset(self) -> bool:
        """Validate PhenoBench dataset."""
        if not self.image_paths:
            self.logger.error("No images found in PhenoBench dataset")
            return False
        
        # Check if required directories exist
        required_dirs = ['PhenoBench/train', 'PhenoBench/val', 'PhenoBench/test']
        for req_dir in required_dirs:
            if not os.path.exists(os.path.join(self.dataset_path, req_dir)):
                self.logger.warning(f"Required directory {req_dir} not found")
        
        return True


class TinyDatasetLoader(DatasetLoader):
    """Loader for TinyDataset."""
    
    def _load_dataset(self):
        """Load TinyDataset structure."""
        self.logger.info(f"Loading TinyDataset from {self.dataset_path}")
        
        images_dir = os.path.join(self.dataset_path, 'Images')
        labels_dir = os.path.join(self.dataset_path, 'Labels')
        
        if os.path.exists(images_dir):
            self.image_paths = get_image_files(images_dir, recursive=False)
        
        if os.path.exists(labels_dir):
            self.annotation_paths = get_image_files(labels_dir, recursive=False)
        
        # Calculate statistics
        self.statistics = calculate_dataset_statistics(self.image_paths)
        self.logger.info(f"Loaded {len(self.image_paths)} images from TinyDataset")
    
    def validate_dataset(self) -> bool:
        """Validate TinyDataset."""
        if not self.image_paths:
            self.logger.error("No images found in TinyDataset")
            return False
        
        if not self.annotation_paths:
            self.logger.warning("No annotations found in TinyDataset")
        
        return True


class WeedAugmentedLoader(DatasetLoader):
    """Loader for Weed Augmented dataset."""
    
    def _load_dataset(self):
        """Load Weed Augmented dataset structure."""
        self.logger.info(f"Loading Weed Augmented dataset from {self.dataset_path}")
        
        images_dir = os.path.join(self.dataset_path, 'images')
        masks_dir = os.path.join(self.dataset_path, 'masks')
        
        if os.path.exists(images_dir):
            self.image_paths = get_image_files(images_dir, recursive=False)
        
        if os.path.exists(masks_dir):
            self.annotation_paths = get_image_files(masks_dir, recursive=False)
        
        # Calculate statistics
        self.statistics = calculate_dataset_statistics(self.image_paths)
        self.logger.info(f"Loaded {len(self.image_paths)} images from Weed Augmented dataset")
    
    def validate_dataset(self) -> bool:
        """Validate Weed Augmented dataset."""
        if not self.image_paths:
            self.logger.error("No images found in Weed Augmented dataset")
            return False
        
        if not self.annotation_paths:
            self.logger.warning("No masks found in Weed Augmented dataset")
        
        return True


class VineyardLoader(DatasetLoader):
    """Loader for Vineyard Canopy dataset."""
    
    def _load_dataset(self):
        """Load Vineyard Canopy dataset structure."""
        self.logger.info(f"Loading Vineyard Canopy dataset from {self.dataset_path}")
        
        images_dir = os.path.join(self.dataset_path, 'Images')
        labels_dir = os.path.join(self.dataset_path, 'Labels')
        
        if os.path.exists(images_dir):
            self.image_paths = get_image_files(images_dir, recursive=False)
        
        if os.path.exists(labels_dir):
            self.annotation_paths = get_image_files(labels_dir, recursive=False)
        
        # Calculate statistics
        self.statistics = calculate_dataset_statistics(self.image_paths)
        self.logger.info(f"Loaded {len(self.image_paths)} images from Vineyard Canopy dataset")
    
    def validate_dataset(self) -> bool:
        """Validate Vineyard Canopy dataset."""
        if not self.image_paths:
            self.logger.error("No images found in Vineyard Canopy dataset")
            return False
        
        if not self.annotation_paths:
            self.logger.warning("No labels found in Vineyard Canopy dataset")
        
        return True


class CapsicumLoader(DatasetLoader):
    """Loader for Capsicum Annuum dataset."""
    
    def _load_dataset(self):
        """Load Capsicum Annuum dataset structure."""
        self.logger.info(f"Loading Capsicum Annuum dataset from {self.dataset_path}")
        
        # Check if data.zip exists and needs extraction
        data_zip = os.path.join(self.dataset_path, 'data.zip')
        if os.path.exists(data_zip):
            self.logger.info("Found data.zip, dataset needs extraction")
            self.dataset_info['needs_extraction'] = True
            self.dataset_info['zip_size'] = os.path.getsize(data_zip)
        else:
            # Look for extracted data
            self._load_extracted_data()
        
        self.logger.info("Capsicum Annuum dataset loaded (may need extraction)")
    
    def _load_extracted_data(self):
        """Load extracted data if available."""
        # Look for common image directories
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
        
        if self.image_paths:
            self.statistics = calculate_dataset_statistics(self.image_paths)
    
    def validate_dataset(self) -> bool:
        """Validate Capsicum Annuum dataset."""
        data_zip = os.path.join(self.dataset_path, 'data.zip')
        if not os.path.exists(data_zip):
            self.logger.warning("data.zip not found in Capsicum Annuum dataset")
            return False
        
        return True


class DatasetLoaderFactory:
    """Factory class for creating appropriate dataset loaders."""
    
    @staticmethod
    def create_loader(dataset_type: str, dataset_path: str, dataset_name: str) -> DatasetLoader:
        """Create appropriate dataset loader based on type."""
        loaders = {
            'phenobench': PhenoBenchLoader,
            'tinydataset': TinyDatasetLoader,
            'weed_augmented': WeedAugmentedLoader,
            'vineyard': VineyardLoader,
            'capsicum': CapsicumLoader
        }
        
        if dataset_type not in loaders:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return loaders[dataset_type](dataset_path, dataset_name)


def load_all_datasets(config: Dict[str, Any]) -> Dict[str, DatasetLoader]:
    """Load all datasets specified in configuration."""
    loaders = {}
    
    for dataset_key, dataset_config in config['datasets'].items():
        try:
            loader = DatasetLoaderFactory.create_loader(
                dataset_key,
                dataset_config['source_path'],
                dataset_config['name']
            )
            loaders[dataset_key] = loader
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_key}: {e}")
    
    return loaders
