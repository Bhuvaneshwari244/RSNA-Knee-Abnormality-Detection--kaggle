"""
Training script for knee abnormality detection
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import *
from src.data.dataset import KneeDataset
from src.data.transforms import get_train_transforms, get_valid_transforms
from src.models.model import create_model
from src.utils.metrics import compute_auc_scores, format_scores


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=MIXED_PRECISION):
            logits = model(images)
            loss = criterion(logits, labels)
        
        # Backward pass
        if MIXED_PRECISION:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            with autocast(enabled=MIXED_PRECISION):
                logits = model(images)
                loss = criterion(logits, labels)
            
            # Get predictions (sigmoid for multi-label)
            preds = torch.sigmoid(logits)
            
            # Store predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    scores = compute_auc_scores(all_labels, all_preds, TARGETS)
    
    return avg_loss, scores


def main():
    """Main training function"""
    print("="*50)
    print("RSNA Knee Abnormality Detection - Training")
    print("="*50)
    
    # Set seed
    set_seed(SEED)
    
    # Device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    print(f"\nLoading data from {DATA_DIR}")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Simple train/val split (replace with proper CV later)
    val_size = int(len(train_df) * 0.2)
    val_df = train_df.iloc[:val_size]
    train_df = train_df.iloc[val_size:]
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    
    # Create datasets
    train_dataset = KneeDataset(
        df=train_df,
        image_dir=TRAIN_DIR,
        targets=TARGETS,
        transform=get_train_transforms(IMAGE_SIZE)
    )
    
    val_dataset = KneeDataset(
        df=val_df,
        image_dir=TRAIN_DIR,
        targets=TARGETS,
        transform=get_valid_transforms(IMAGE_SIZE)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model: {MODEL_NAME}")
    model = create_model(
        model_name=MODEL_NAME,
        num_classes=len(TARGETS),
        pretrained=PRETRAINED
    )
    model = model.to(device)
    
    # Loss function (BCEWithLogitsLoss for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=MIXED_PRECISION)
    
    # Training loop
    best_score = 0.0
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs")
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, scores = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}\n")
        print(format_scores(scores))
        
        # Update learning rate
        scheduler.step(scores['macro_auc'])
        
        # Save best model
        if scores['macro_auc'] > best_score:
            best_score = scores['macro_auc']
            save_path = MODELS_DIR / f"{MODEL_NAME}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': best_score,
            }, save_path)
            print(f"\nSaved best model to {save_path}")
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best macro AUC: {best_score:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
