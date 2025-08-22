"""
Utility functions for dataset combination operations.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image
import hashlib
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("dataset_combination")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """Get file extension from file path."""
    return Path(file_path).suffix.lower()


def is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return get_file_extension(file_path) in image_extensions


def get_image_files(directory: str, recursive: bool = True) -> List[str]:
    """Get all image files from directory."""
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if is_image_file(file):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if is_image_file(file):
                image_files.append(os.path.join(directory, file))
    return sorted(image_files)


def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_image_file(file_path: str) -> bool:
    """Validate if image file is readable and not corrupted."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(file_path: str) -> Dict[str, Any]:
    """Get basic information about an image file."""
    try:
        with Image.open(file_path) as img:
            info = {
                'path': file_path,
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'file_size': os.path.getsize(file_path),
                'hash': calculate_file_hash(file_path)
            }
        return info
    except Exception as e:
        return {
            'path': file_path,
            'error': str(e)
        }


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                method: str = 'bilinear') -> np.ndarray:
    """Resize image to target size."""
    if method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR
    
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def create_annotation_mapping(source_labels: List[int], target_labels: List[int]) -> Dict[int, int]:
    """Create mapping between source and target label sets."""
    if len(source_labels) != len(target_labels):
        raise ValueError("Source and target label lists must have same length")
    
    return dict(zip(source_labels, target_labels))


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """Save metadata to JSON file."""
    ensure_directory(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_dataset_info(dataset_name: str, source_path: str, 
                       num_images: int, num_annotations: int,
                       image_size: Tuple[int, int], classes: List[str]) -> Dict[str, Any]:
    """Create dataset information dictionary."""
    return {
        'name': dataset_name,
        'source_path': source_path,
        'num_images': num_images,
        'num_annotations': num_annotations,
        'image_size': image_size,
        'classes': classes,
        'created_at': datetime.now().isoformat(),
        'version': '1.0.0'
    }


def calculate_dataset_statistics(image_paths: List[str], 
                               annotation_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate basic statistics for a dataset."""
    stats = {
        'total_images': len(image_paths),
        'image_sizes': [],
        'file_sizes': [],
        'formats': {},
        'total_size_mb': 0
    }
    
    for img_path in image_paths:
        try:
            # Get image size
            with Image.open(img_path) as img:
                stats['image_sizes'].append(img.size)
            
            # Get file size
            file_size = os.path.getsize(img_path)
            stats['file_sizes'].append(file_size)
            stats['total_size_mb'] += file_size / (1024 * 1024)
            
            # Count formats
            ext = get_file_extension(img_path)
            stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
            
        except Exception as e:
            logging.warning(f"Error processing {img_path}: {e}")
    
    if stats['image_sizes']:
        stats['min_size'] = min(stats['image_sizes'])
        stats['max_size'] = max(stats['image_sizes'])
        stats['avg_size'] = tuple(np.mean(stats['image_sizes'], axis=0).astype(int))
    
    if stats['file_sizes']:
        stats['min_file_size'] = min(stats['file_sizes'])
        stats['max_file_size'] = max(stats['file_sizes'])
        stats['avg_file_size'] = np.mean(stats['file_sizes'])
    
    return stats


def create_progress_tracker(total_items: int, description: str = "Processing"):
    """Create a progress tracker for long operations."""
    from tqdm import tqdm
    return tqdm(total=total_items, desc=description, unit="items")


def backup_file(file_path: str, backup_dir: str) -> str:
    """Create a backup of a file."""
    ensure_directory(backup_dir)
    filename = os.path.basename(file_path)
    backup_path = os.path.join(backup_dir, f"backup_{filename}")
    
    if os.path.exists(file_path):
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    return ""


def cleanup_temp_files(temp_dir: str) -> None:
    """Clean up temporary files and directories."""
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
