#!/usr/bin/env python3
"""
Update Metadata with Real Fruit Data
Updates the main dataset metadata with the integrated fruit datasets
"""

import json
from pathlib import Path
from datetime import datetime

def update_metadata():
    """Update the main dataset metadata with fruit data"""
    base_path = Path("/Volumes/Rapid/Agriculture Dataset")
    metadata_path = base_path / "Combined_datasets" / "metadata.json"
    
    # Load existing metadata
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Count fruit images
    fruit_datasets_path = base_path / "Combined_datasets" / "fruit_datasets"
    total_fruit_images = 0
    fruit_datasets_info = {}
    
    if fruit_datasets_path.exists():
        for dataset_dir in fruit_datasets_path.iterdir():
            if dataset_dir.is_dir():
                # Count images in this dataset
                image_count = len(list(dataset_dir.rglob("*.png")))
                total_fruit_images += image_count
                
                # Get class information
                classes = set()
                for img_file in dataset_dir.rglob("*.png"):
                    if "mask" not in img_file.name:
                        class_name = img_file.stem.split('_')[-1]
                        classes.add(class_name)
                
                fruit_datasets_info[dataset_dir.name] = {
                    "image_count": image_count,
                    "class_count": len(classes),
                    "classes": sorted(list(classes))
                }
    
    # Update metadata
    metadata["fruit_integration"] = {
        "integration_date": datetime.now().isoformat(),
        "total_fruit_images": total_fruit_images,
        "fruit_datasets": fruit_datasets_info,
        "total_fruit_classes": sum(info["class_count"] for info in fruit_datasets_info.values())
    }
    
    # Update total counts
    if "total_images" in metadata:
        metadata["total_images"] += total_fruit_images
    else:
        metadata["total_images"] = total_fruit_images
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Updated metadata with {total_fruit_images:,} fruit images")
    print(f"📊 Fruit datasets: {len(fruit_datasets_info)}")
    for name, info in fruit_datasets_info.items():
        print(f"  {name}: {info['image_count']:,} images, {info['class_count']} classes")

if __name__ == "__main__":
    update_metadata()
