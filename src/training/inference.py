"""
Inference script for generating submissions
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import *
from src.data.dataset import KneeDataset
from src.data.transforms import get_valid_transforms, get_tta_transforms
from src.models.model import create_model


def inference(model, loader, device, use_tta=False):
    """
    Run inference on test set
    
    Args:
        model: Trained model
        loader: DataLoader for test set
        device: Device to run on
        use_tta: Whether to use test-time augmentation
        
    Returns:
        Array of predictions (N, num_classes)
        List of study IDs
    """
    model.eval()
    all_preds = []
    all_study_ids = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference")
        for batch in pbar:
            images = batch['image'].to(device)
            study_ids = batch['study_id']
            
            # Forward pass
            logits = model(images)
            preds = torch.sigmoid(logits)
            
            all_preds.append(preds.cpu().numpy())
            all_study_ids.extend(study_ids)
    
    # Concatenate predictions
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_preds, all_study_ids


def create_submission(
    model_path: Path,
    test_csv: Path,
    output_path: Path,
    use_tta: bool = False
):
    """
    Create submission file
    
    Args:
        model_path: Path to trained model checkpoint
        test_csv: Path to test.csv (if exists) or create from test directory
        output_path: Path to save submission.csv
        use_tta: Whether to use test-time augmentation
    """
    print("="*50)
    print("RSNA Knee Abnormality Detection - Inference")
    print("="*50)
    
    # Device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = create_model(
        model_name=MODEL_NAME,
        num_classes=len(TARGETS),
        pretrained=False
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded (best score: {checkpoint.get('score', 'N/A')})")
    
    # Load test data
    # Note: Adjust based on actual test data structure
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
    else:
        # Create test_df from directory listing
        print(f"Creating test dataframe from {TEST_DIR}")
        test_files = list(TEST_DIR.glob("*.dcm"))
        test_df = pd.DataFrame({
            'StudyInstanceUID': [f.stem for f in test_files]
        })
    
    print(f"Test samples: {len(test_df)}")
    
    # Create dataset
    test_dataset = KneeDataset(
        df=test_df,
        image_dir=TEST_DIR,
        targets=TARGETS,
        transform=get_valid_transforms(IMAGE_SIZE),
        is_test=True
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Run inference
    print("\nRunning inference...")
    predictions, study_ids = inference(model, test_loader, device, use_tta)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'StudyInstanceUID': study_ids
    })
    
    # Add predictions for each target
    for i, target in enumerate(TARGETS):
        submission[target] = predictions[:, i]
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"\nFirst few rows:")
    print(submission.head())
    
    print(f"\n{'='*50}")
    print("Inference complete!")
    print(f"{'='*50}")
    
    return submission


def main():
    """Main function"""
    # Paths
    model_path = MODELS_DIR / f"{MODEL_NAME}_best.pth"
    test_csv = DATA_DIR / "test.csv"
    output_path = SUBMISSIONS_DIR / "submission.csv"
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Create submission
    create_submission(
        model_path=model_path,
        test_csv=test_csv,
        output_path=output_path,
        use_tta=TEST_TIME_AUGMENTATION
    )


if __name__ == "__main__":
    main()
