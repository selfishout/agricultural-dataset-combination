"""
Training script for the U-Net segmentation model using the combined agricultural dataset.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from dataset import create_data_loaders
from model import create_model, create_loss_function, create_optimizer, create_scheduler

class Trainer:
    """Training class for the segmentation model."""
    
    def __init__(self):
        """Initialize the trainer."""
        self.device = torch.device(config.DEVICE)
        print(f"🚀 Training on device: {self.device}")
        
        # Create directories
        config.create_directories()
        
        # Create model, loss, optimizer, and scheduler
        self.model = create_model(self.device)
        self.criterion = create_loss_function()
        self.optimizer = create_optimizer(self.model)
        self.scheduler = create_scheduler(self.optimizer)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        self.best_val_dice = 0.0
        
        print(f"✅ Trainer initialized successfully!")
    
    def calculate_dice_score(self, predictions, targets):
        """Calculate Dice score for evaluation."""
        predictions = torch.sigmoid(predictions) > 0.5
        targets = targets > 0.5
        
        intersection = (predictions & targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection) / (union + 1e-6)
        return dice.item()
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss/(batch_idx+1):.4f}"
            })
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), 
                                     epoch * num_batches + batch_idx)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Get data
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate Dice score
                dice_score = self.calculate_dice_score(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                total_dice += dice_score
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Dice': f"{dice_score:.4f}",
                    'Avg Dice': f"{total_dice/(batch_idx+1):.4f}"
                })
                
                # Log to TensorBoard
                if batch_idx % 5 == 0:
                    self.writer.add_scalar('Val/Batch_Loss', loss.item(), 
                                         epoch * num_batches + batch_idx)
                    self.writer.add_scalar('Val/Batch_Dice', dice_score, 
                                         epoch * num_batches + batch_idx)
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        self.val_losses.append(avg_loss)
        self.val_dice_scores.append(avg_dice)
        
        return avg_loss, avg_dice
    
    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_dice': val_dice,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores,
            'best_val_dice': self.best_val_dice
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f"🏆 New best model saved! Dice Score: {val_dice:.4f}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Dice scores
        ax2.plot(self.val_dice_scores, label='Validation Dice Score', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Validation Dice Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting training for {config.NUM_EPOCHS} epochs...")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"📊 Test samples: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_dice = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            self.writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Epoch_Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Epoch_Dice', val_dice, epoch)
            self.writer.add_scalar('Train/Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Training Loss: {train_loss:.4f}")
            print(f"   Validation Loss: {val_loss:.4f}")
            print(f"   Validation Dice: {val_dice:.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
            
            self.save_checkpoint(epoch, val_dice, is_best)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/60:.2f} minutes!")
        print(f"🏆 Best validation Dice score: {self.best_val_dice:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.best_val_dice

def main():
    """Main function to run training."""
    print("🌾 Agricultural Image Segmentation Training")
    print("=" * 60)
    
    try:
        # Validate dataset paths
        config.validate_paths()
        
        # Create and run trainer
        trainer = Trainer()
        best_dice = trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Best validation Dice score: {best_dice:.4f}")
        print(f"📁 Checkpoints saved in: {config.CHECKPOINT_DIR}")
        print(f"📊 Logs saved in: {config.LOG_DIR}")
        print(f"📈 Results saved in: {config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
