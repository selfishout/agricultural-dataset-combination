"""
Dataset loader for the Combined Agricultural Dataset.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import config

class AgriculturalSegmentationDataset(Dataset):
    """Dataset class for agricultural image segmentation."""
    
    def __init__(self, images_dir, annotations_dir, transform=None, is_training=True):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing annotations
            transform: Albumentations transforms
            is_training: Whether this is for training (affects augmentation)
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.is_training = is_training
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.endswith('.png') and not f.startswith('.')]
        
        print(f"📁 Found {len(self.image_files)} images in {images_dir}")
        
        # Validate that annotations exist for all images
        self.valid_pairs = []
        for img_file in self.image_files:
            ann_file = img_file  # Same filename for annotations
            ann_path = os.path.join(annotations_dir, ann_file)
            
            if os.path.exists(ann_path):
                self.valid_pairs.append((img_file, ann_file))
            else:
                print(f"⚠️ Warning: No annotation found for {img_file}")
        
        print(f"✅ Valid image-annotation pairs: {len(self.valid_pairs)}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        img_file, ann_file = self.valid_pairs[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load annotation
        ann_path = os.path.join(self.annotations_dir, ann_file)
        annotation = Image.open(ann_path).convert('L')
        annotation = np.array(annotation)
        
        # Normalize annotation to binary (0 or 1)
        annotation = (annotation > 0).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=annotation)
            image = transformed['image']
            annotation = transformed['mask']
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(annotation, np.ndarray):
            annotation = torch.from_numpy(annotation).unsqueeze(0).float()
        
        return {
            'image': image,
            'mask': annotation,
            'filename': img_file
        }

def get_transforms(input_size=(512, 512)):
    """
    Get data augmentation transforms.
    
    Args:
        input_size: Target image size (height, width)
    
    Returns:
        train_transforms: Transforms for training
        val_transforms: Transforms for validation/testing
    """
    
    # Training transforms with augmentation
    train_transforms = A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Validation/testing transforms (no augmentation)
    val_transforms = A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transforms, val_transforms

def create_data_loaders(batch_size=8, num_workers=4):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Testing data loader
    """
    
    # Get transforms
    train_transforms, val_transforms = get_transforms(config.INPUT_SIZE)
    
    # Create datasets
    train_dataset = AgriculturalSegmentationDataset(
        images_dir=config.TRAIN_IMAGES,
        annotations_dir=config.TRAIN_ANNOTATIONS,
        transform=train_transforms,
        is_training=True
    )
    
    val_dataset = AgriculturalSegmentationDataset(
        images_dir=config.VAL_IMAGES,
        annotations_dir=config.VAL_ANNOTATIONS,
        transform=val_transforms,
        is_training=False
    )
    
    test_dataset = AgriculturalSegmentationDataset(
        images_dir=config.TEST_IMAGES,
        annotations_dir=config.TEST_ANNOTATIONS,
        transform=val_transforms,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"📊 Data loaders created:")
    print(f"   Training: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"   Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"   Testing: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader

def visualize_sample_batch(data_loader, num_samples=4):
    """
    Visualize a sample batch from the data loader.
    
    Args:
        data_loader: Data loader to sample from
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    # Get a batch
    batch = next(iter(data_loader))
    images = batch['image'][:num_samples]
    masks = batch['mask'][:num_samples]
    filenames = batch['filename'][:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    fig.suptitle('Sample Batch from Combined Agricultural Dataset', fontsize=16)
    
    for i in range(num_samples):
        # Show image
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # Denormalize
        img = np.clip(img, 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image: {filenames[i][:20]}...', fontsize=10)
        axes[0, i].axis('off')
        
        # Show mask
        mask = masks[i].squeeze().numpy()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Mask: {filenames[i][:20]}...', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_batch_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample batch visualization saved as 'sample_batch_visualization.png'")

if __name__ == "__main__":
    # Test the dataset
    print("🧪 Testing dataset loader...")
    
    try:
        # Validate paths
        config.validate_paths()
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=4, num_workers=0
        )
        
        # Visualize sample batch
        visualize_sample_batch(train_loader, num_samples=4)
        
        print("✅ Dataset loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing dataset loader: {e}")
        import traceback
        traceback.print_exc()
