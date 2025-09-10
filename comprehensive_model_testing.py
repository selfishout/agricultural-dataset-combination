#!/usr/bin/env python3
"""
Comprehensive Model Testing Framework
Tests the agricultural dataset with various segmentation models
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path
sys.path.append('src')

from utils import setup_logging, load_config

class AgriculturalDataset(Dataset):
    """PyTorch Dataset for agricultural segmentation"""
    
    def __init__(self, images_dir, masks_dir, transform=None, subset_size=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith('.png') and not f.startswith('.')]
        
        if subset_size:
            self.image_files = self.image_files[:subset_size]
        
        print(f"📊 Dataset loaded: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load mask
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
        else:
            # Create dummy mask if not found
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask

class SimpleUNet(nn.Module):
    """Simple U-Net architecture for segmentation"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(torch.max_pool2d(enc1, 2))
        enc3 = self.enc3(torch.max_pool2d(enc2, 2))
        enc4 = self.enc4(torch.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(torch.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final layer
        output = self.final(dec1)
        return torch.sigmoid(output)

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class ModelTester:
    """Comprehensive model testing framework"""
    
    def __init__(self, config_path="config/dataset_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging("INFO")
        
        # Setup directories
        self.testing_dir = "model_testing_results"
        os.makedirs(self.testing_dir, exist_ok=True)
        
        # Model configurations
        self.models = {
            "SimpleUNet": {
                "class": SimpleUNet,
                "params": {"in_channels": 3, "out_channels": 1},
                "description": "Simple U-Net architecture"
            }
        }
        
        # Training configurations
        self.training_configs = {
            "quick_test": {
                "epochs": 2,
                "batch_size": 4,
                "subset_size": 100,
                "description": "Quick test with minimal data"
            },
            "standard_test": {
                "epochs": 5,
                "batch_size": 8,
                "subset_size": 500,
                "description": "Standard test with moderate data"
            },
            "comprehensive_test": {
                "epochs": 10,
                "batch_size": 16,
                "subset_size": 1000,
                "description": "Comprehensive test with more data"
            }
        }
    
    def create_data_loaders(self, dataset_path, config):
        """Create data loaders for training and validation"""
        print(f"📊 Creating data loaders with config: {config['description']}")
        
        # Define transforms
        train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Create datasets
        train_dataset = AgriculturalDataset(
            os.path.join(dataset_path, "train", "images"),
            os.path.join(dataset_path, "train", "masks"),
            transform=train_transform,
            subset_size=config['subset_size']
        )
        
        val_dataset = AgriculturalDataset(
            os.path.join(dataset_path, "val", "images"),
            os.path.join(dataset_path, "val", "masks"),
            transform=val_transform,
            subset_size=config['subset_size'] // 4
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def train_model(self, model, train_loader, val_loader, config, model_name):
        """Train a model and return training history"""
        print(f"🚀 Training {model_name} with {config['description']}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Loss and optimizer
        criterion = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'epochs': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_dice = 0.0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)
                masks = masks.float().unsqueeze(1) / 255.0  # Normalize masks
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate dice score
                with torch.no_grad():
                    preds = (outputs > 0.5).float()
                    dice = self.calculate_dice_score(preds, masks)
                    train_dice += dice
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    masks = masks.float().unsqueeze(1) / 255.0
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    dice = self.calculate_dice_score(preds, masks)
                    val_dice += dice
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_dice = train_dice / len(train_loader)
            avg_val_dice = val_dice / len(val_loader)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_dice'].append(avg_train_dice)
            history['val_dice'].append(avg_val_dice)
            history['epochs'].append(epoch + 1)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, history
    
    def calculate_dice_score(self, preds, targets):
        """Calculate Dice score"""
        smooth = 1.0
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        
        return dice.item()
    
    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate model performance"""
        print(f"📊 Evaluating {model_name}...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_targets = []
        dice_scores = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.float().unsqueeze(1) / 255.0
                
                outputs = model(images)
                preds = (outputs > 0.5).float()
                
                # Calculate dice score for this batch
                dice = self.calculate_dice_score(preds, masks)
                dice_scores.append(dice)
                
                # Store predictions and targets
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Convert to binary for classification metrics
        preds_binary = (all_preds > 0.5).astype(int)
        targets_binary = (all_targets > 0.5).astype(int)
        
        metrics = {
            'dice_score': np.mean(dice_scores),
            'accuracy': accuracy_score(targets_binary, preds_binary),
            'precision': precision_score(targets_binary, preds_binary, zero_division=0),
            'recall': recall_score(targets_binary, preds_binary, zero_division=0),
            'f1_score': f1_score(targets_binary, preds_binary, zero_division=0),
            'mean_dice_per_batch': dice_scores
        }
        
        return metrics, all_preds, all_targets
    
    def create_visualizations(self, history, metrics, model_name, config_name):
        """Create training and evaluation visualizations"""
        print(f"📊 Creating visualizations for {model_name}...")
        
        viz_dir = os.path.join(self.testing_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} - {config_name} Training Results', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history['epochs'], history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['epochs'], history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice score plot
        axes[0, 1].plot(history['epochs'], history['train_dice'], label='Train Dice', color='blue')
        axes[0, 1].plot(history['epochs'], history['val_dice'], label='Val Dice', color='red')
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Metrics bar chart
        metric_names = ['Dice Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            metrics['dice_score'],
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        bars = axes[1, 0].bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Dice score distribution
        axes[1, 1].hist(metrics['mean_dice_per_batch'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Dice Score Distribution (Per Batch)')
        axes[1, 1].set_xlabel('Dice Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(metrics['dice_score'], color='red', linestyle='--', 
                          label=f'Mean: {metrics["dice_score"]:.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{model_name}_{config_name}_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved: {viz_dir}")
    
    def run_comprehensive_testing(self, dataset_path):
        """Run comprehensive testing on all models and configurations"""
        print("🧪 COMPREHENSIVE MODEL TESTING")
        print("=" * 60)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return None
        
        # Test results storage
        all_results = {
            'dataset_path': dataset_path,
            'testing_date': datetime.now().isoformat(),
            'models_tested': {},
            'overall_summary': {}
        }
        
        # Test each model with each configuration
        for model_name, model_config in self.models.items():
            print(f"\n🤖 Testing Model: {model_name}")
            print(f"   Description: {model_config['description']}")
            
            all_results['models_tested'][model_name] = {}
            
            for config_name, config in self.training_configs.items():
                print(f"\n📊 Configuration: {config_name}")
                print(f"   Description: {config['description']}")
                print(f"   Epochs: {config['epochs']}, Batch Size: {config['batch_size']}")
                
                try:
                    # Create data loaders
                    train_loader, val_loader = self.create_data_loaders(dataset_path, config)
                    
                    # Create test loader (use validation data for testing)
                    test_loader = val_loader
                    
                    # Initialize model
                    model = model_config['class'](**model_config['params'])
                    
                    # Train model
                    start_time = time.time()
                    trained_model, history = self.train_model(model, train_loader, val_loader, config, model_name)
                    training_time = time.time() - start_time
                    
                    # Evaluate model
                    metrics, preds, targets = self.evaluate_model(trained_model, test_loader, model_name)
                    
                    # Create visualizations
                    self.create_visualizations(history, metrics, model_name, config_name)
                    
                    # Store results
                    config_results = {
                        'config': config,
                        'training_time': training_time,
                        'history': history,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    
                    all_results['models_tested'][model_name][config_name] = config_results
                    
                    print(f"✅ {model_name} - {config_name} completed successfully")
                    print(f"   Training time: {training_time:.2f} seconds")
                    print(f"   Final Dice Score: {metrics['dice_score']:.4f}")
                    print(f"   Final Accuracy: {metrics['accuracy']:.4f}")
                    
                except Exception as e:
                    print(f"❌ {model_name} - {config_name} failed: {e}")
                    all_results['models_tested'][model_name][config_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        # Create overall summary
        self.create_overall_summary(all_results)
        
        # Save results
        results_file = os.path.join(self.testing_dir, "comprehensive_testing_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📊 COMPREHENSIVE TESTING COMPLETE")
        print(f"   Results saved: {results_file}")
        print(f"   Visualizations: {os.path.join(self.testing_dir, 'visualizations')}")
        
        return all_results
    
    def create_overall_summary(self, results):
        """Create overall summary of testing results"""
        print("📊 Creating overall summary...")
        
        summary = {
            'total_models': len(results['models_tested']),
            'total_configs': len(self.training_configs),
            'successful_tests': 0,
            'failed_tests': 0,
            'best_performances': {},
            'average_metrics': {}
        }
        
        all_dice_scores = []
        all_accuracies = []
        
        for model_name, model_results in results['models_tested'].items():
            model_dice_scores = []
            model_accuracies = []
            
            for config_name, config_results in model_results.items():
                if config_results.get('status') == 'success':
                    summary['successful_tests'] += 1
                    metrics = config_results['metrics']
                    model_dice_scores.append(metrics['dice_score'])
                    model_accuracies.append(metrics['accuracy'])
                    all_dice_scores.append(metrics['dice_score'])
                    all_accuracies.append(metrics['accuracy'])
                else:
                    summary['failed_tests'] += 1
            
            if model_dice_scores:
                summary['best_performances'][model_name] = {
                    'best_dice': max(model_dice_scores),
                    'best_accuracy': max(model_accuracies),
                    'avg_dice': np.mean(model_dice_scores),
                    'avg_accuracy': np.mean(model_accuracies)
                }
        
        if all_dice_scores:
            summary['average_metrics'] = {
                'overall_avg_dice': np.mean(all_dice_scores),
                'overall_avg_accuracy': np.mean(all_accuracies),
                'best_dice_overall': max(all_dice_scores),
                'best_accuracy_overall': max(all_accuracies)
            }
        
        results['overall_summary'] = summary
        
        # Print summary
        print(f"\n📊 OVERALL TESTING SUMMARY")
        print(f"=" * 50)
        print(f"Total models tested: {summary['total_models']}")
        print(f"Total configurations: {summary['total_configs']}")
        print(f"Successful tests: {summary['successful_tests']}")
        print(f"Failed tests: {summary['failed_tests']}")
        
        if summary['average_metrics']:
            print(f"Overall average Dice score: {summary['average_metrics']['overall_avg_dice']:.4f}")
            print(f"Overall average accuracy: {summary['average_metrics']['overall_avg_accuracy']:.4f}")
            print(f"Best Dice score overall: {summary['average_metrics']['best_dice_overall']:.4f}")
            print(f"Best accuracy overall: {summary['average_metrics']['best_accuracy_overall']:.4f}")

def main():
    """Main testing function"""
    print("🧪 COMPREHENSIVE MODEL TESTING FRAMEWORK")
    print("=" * 60)
    
    # Initialize tester
    tester = ModelTester()
    
    # Check if dataset exists - try multiple paths
    possible_paths = [
        "/Volumes/Rapid/Agriculture Dataset/Combined_datasets",
        "sample_segmentation_project/demo_dataset",
        "data/combined_datasets"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Found dataset at: {path}")
            break
    
    if not dataset_path:
        print(f"❌ No dataset found at any of these paths:")
        for path in possible_paths:
            print(f"   - {path}")
        print("Please ensure a dataset is available.")
        return False
    
    # Run comprehensive testing
    results = tester.run_comprehensive_testing(dataset_path)
    
    if results:
        print("\n✅ Comprehensive testing completed successfully!")
        return True
    else:
        print("\n❌ Comprehensive testing failed!")
        return False

if __name__ == "__main__":
    main()
