"""
Evaluation script for the trained U-Net segmentation model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import config
from dataset import create_data_loaders
from model import create_model

class ModelEvaluator:
    """Class for evaluating the trained segmentation model."""
    
    def __init__(self, checkpoint_path=None):
        """Initialize the evaluator."""
        self.device = torch.device(config.DEVICE)
        print(f"🔍 Evaluating on device: {self.device}")
        
        # Create model
        self.model = create_model(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            # Try to load best model
            best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            if os.path.exists(best_model_path):
                self.load_checkpoint(best_model_path)
                print(f"✅ Loaded best model from: {best_model_path}")
            else:
                print("⚠️ No checkpoint found. Using untrained model.")
        
        # Create data loaders
        _, _, self.test_loader = create_data_loaders(
            batch_size=config.EVAL_BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )
        
        print(f"✅ Evaluator initialized successfully!")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch'] + 1}")
        print(f"   Validation Dice: {checkpoint['val_dice']:.4f}")
    
    def calculate_metrics(self, predictions, targets):
        """Calculate evaluation metrics."""
        # Convert to binary
        predictions = (predictions > 0.5).astype(np.uint8)
        targets = (targets > 0.5).astype(np.uint8)
        
        # Flatten for metric calculation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(target_flat, pred_flat)
        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        
        # Calculate IoU (Intersection over Union)
        intersection = np.logical_and(predictions, targets).sum()
        union = np.logical_or(predictions, targets).sum()
        iou = intersection / (union + 1e-6)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection) / (predictions.sum() + targets.sum() + 1e-6)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'dice': dice
        }
    
    def evaluate(self):
        """Evaluate the model on test set."""
        self.model.eval()
        
        all_metrics = []
        all_predictions = []
        all_targets = []
        sample_images = []
        sample_masks = []
        sample_preds = []
        
        print(f"🔍 Evaluating model on test set...")
        progress_bar = tqdm(self.test_loader, desc="Evaluating")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Get data
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                # Convert to numpy for metric calculation
                pred_np = predictions.cpu().numpy()
                mask_np = masks.cpu().numpy()
                
                # Calculate metrics for each sample in batch
                for i in range(pred_np.shape[0]):
                    metrics = self.calculate_metrics(pred_np[i], mask_np[i])
                    all_metrics.append(metrics)
                    
                    # Store predictions and targets
                    all_predictions.append(pred_np[i])
                    all_targets.append(mask_np[i])
                    
                    # Store samples for visualization
                    if len(sample_images) < config.NUM_SAMPLES_TO_VISUALIZE:
                        sample_images.append(images[i].cpu())
                        sample_masks.append(masks[i].cpu())
                        sample_preds.append(predictions[i].cpu())
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    avg_dice = np.mean([m['dice'] for m in all_metrics])
                    progress_bar.set_postfix({'Avg Dice': f"{avg_dice:.4f}"})
        
        # Calculate overall metrics
        overall_metrics = {}
        for key in all_metrics[0].keys():
            overall_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Print results
        print(f"\n📊 Evaluation Results:")
        print(f"{'='*50}")
        for metric, value in overall_metrics.items():
            print(f"   {metric.capitalize():12}: {value:.4f}")
        
        # Save results
        if config.SAVE_PREDICTIONS:
            self.save_predictions(all_predictions, all_targets)
        
        # Visualize sample results
        if config.SAVE_VISUALIZATIONS:
            self.visualize_results(sample_images, sample_masks, sample_preds)
        
        return overall_metrics, all_metrics
    
    def save_predictions(self, predictions, targets):
        """Save predictions and targets."""
        results_dir = os.path.join(config.RESULTS_DIR, 'predictions')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save as numpy arrays
        np.save(os.path.join(results_dir, 'predictions.npy'), np.array(predictions))
        np.save(os.path.join(results_dir, 'targets.npy'), np.array(targets))
        
        print(f"✅ Predictions saved to: {results_dir}")
    
    def visualize_results(self, images, masks, predictions):
        """Visualize sample results."""
        num_samples = min(len(images), config.NUM_SAMPLES_TO_VISUALIZE)
        
        fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
        fig.suptitle('Sample Segmentation Results', fontsize=16)
        
        for i in range(num_samples):
            # Original image
            img = images[i].permute(1, 2, 0).numpy()
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # Denormalize
            img = np.clip(img, 0, 1)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Original Image {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # Ground truth mask
            mask = masks[i].squeeze().numpy()
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}', fontsize=10)
            axes[1, i].axis('off')
            
            # Prediction
            pred = predictions[i].squeeze().numpy()
            axes[2, i].imshow(pred, cmap='gray')
            axes[2, i].set_title(f'Prediction {i+1}', fontsize=10)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'sample_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Sample results visualization saved!")
    
    def create_confusion_matrix(self, predictions, targets):
        """Create and visualize confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        # Convert to binary
        pred_binary = (np.array(predictions) > 0.5).flatten()
        target_binary = (np.array(targets) > 0.5).flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(target_binary, pred_binary)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved!")

def main():
    """Main function to run evaluation."""
    print("🔍 Agricultural Image Segmentation Model Evaluation")
    print("=" * 60)
    
    try:
        # Validate dataset paths
        config.validate_paths()
        
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Run evaluation
        overall_metrics, all_metrics = evaluator.evaluate()
        
        # Print final summary
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Overall Performance:")
        for metric, value in overall_metrics.items():
            print(f"   {metric.capitalize():12}: {value:.4f}")
        
        # Create confusion matrix
        if config.SAVE_VISUALIZATIONS:
            evaluator.create_confusion_matrix(
                [m['dice'] for m in all_metrics],
                [m['accuracy'] for m in all_metrics]
            )
        
        print(f"\n📁 Results saved in: {config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
