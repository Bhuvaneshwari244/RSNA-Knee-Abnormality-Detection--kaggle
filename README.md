# RSNA Knee Abnormality Detection - Kaggle Competition

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)

A complete machine learning pipeline for detecting 12 knee abnormalities from MRI scans using deep learning.

## 📋 Competition Overview

**Competition:** RSNA Knee Abnormality Detection  
**Host:** Radiological Society of North America  
**Task:** Multi-label classification of knee MRI scans  
**Metric:** Macro-averaged AUC ROC  
**Prize Pool:** $77,000  
**Deadline:** October 22, 2026

### Target Abnormalities (12 classes):
1. ACL (Anterior Cruciate Ligament)
2. MCL (Medial Collateral Ligament)
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA (Osteoarthritis)
6. Lateral OA
7. PF OA (Patellofemoral OA)
8. Effusion
9. Synovitis
10. Baker's Cyst
11. Contusion
12. Fracture

## 🚀 Quick Start

### 1. Upload to Kaggle

**Main Notebook:** `download.ipynb`

1. Go to [Competition Code Page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code)
2. Click "New Notebook" → "File" → "Upload Notebook"
3. Upload `download.ipynb`
4. Add Competition Data:
   - Click "+ Add Input"
   - Search "rsna-knee-abnormality-detection"
   - Click "Add"
5. Enable GPU:
   - Settings → Accelerator → GPU T4 x2
6. Click "Run All"
7. Wait 30-60 minutes
8. Download `submission.csv` from Output

### 2. Submit Results

1. Go to [Submit Page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submit)
2. Upload `submission.csv`
3. View your score on the leaderboard!

## 📁 Project Structure

```
RSNA-Knee-Abnormality-Detection/
├── download.ipynb              # ⭐ MAIN NOTEBOOK - Upload this to Kaggle
├── RSNA_Fixed.ipynb            # Alternative notebook with fixes
├── RSNA_Fixed_V2.ipynb         # Backup version
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── data/                       # Competition data (not in repo)
├── models/                     # Saved model weights
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code
│   ├── data/                   # Dataset classes
│   ├── models/                 # Model architectures
│   ├── training/               # Training scripts
│   └── utils/                  # Utility functions
└── submissions/                # Generated submission files
```

## 🛠️ Installation (Local Development)

```bash
# Clone repository
git clone https://github.com/Bhuvaneshwari244/RSNA-Knee-Abnormality-Detection--kaggle.git
cd RSNA-Knee-Abnormality-Detection--kaggle

# Install dependencies
pip install -r requirements.txt
```

## 📊 Model Architecture

**Backbone:** EfficientNet-B0 (pretrained on ImageNet)

- **Input:** MRI DICOM images (224x224x3)
- **Output:** 12 binary predictions (one per abnormality)
- **Loss Function:** BCEWithLogitsLoss (multi-label classification)
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau

## 🔧 Configuration

Edit `config.py` to customize:

```python
MODEL_NAME = 'efficientnet_b0'  # Try: efficientnet_b1, resnet50, etc.
IMAGE_SIZE = 224                 # Image resolution
BATCH_SIZE = 16                  # Batch size
NUM_EPOCHS = 3                   # Training epochs
LEARNING_RATE = 1e-4             # Learning rate
```

## 📈 Training Pipeline

1. **Data Loading**: DICOM image preprocessing
2. **Data Augmentation**: Flips, rotations, brightness adjustments
3. **Model Training**: EfficientNet with transfer learning
4. **Validation**: 80/20 train-val split
5. **Inference**: Generate predictions on test set
6. **Submission**: Create submission.csv

## 🎯 Results

**Current Performance:**
- Validation AUC: 0.5000 (baseline)
- Status: Model completes training but needs improved DICOM preprocessing

**Known Issues:**
- DICOM image loading needs optimization for multi-series MRI data
- Loss becomes NaN due to complex nested folder structure
- Requires better handling of 3D volumetric data

## 🔍 Key Features

✅ **Complete Pipeline**: End-to-end from data loading to submission  
✅ **GPU Optimized**: Mixed precision training with AMP  
✅ **Data Augmentation**: Comprehensive image augmentations  
✅ **Transfer Learning**: Pretrained EfficientNet backbone  
✅ **Progress Tracking**: tqdm progress bars for all operations  
✅ **Model Checkpointing**: Saves best model based on validation AUC  

## 🚧 Improvements Needed

1. **Better DICOM Handling**
   - Handle multiple series per study
   - Process 3D volumetric data properly
   - Implement proper medical image preprocessing

2. **Multi-Modal Learning**
   - Use radiology reports (text data)
   - Combine image + text features

3. **Advanced Techniques**
   - Cross-validation (5-fold)
   - Ensemble multiple models
   - Test-time augmentation

4. **Optimization**
   - Larger models (EfficientNet-B2, B3)
   - Higher resolution images (384x384)
   - More training epochs (10-20)

## 📚 Resources

- [Competition Page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [Dataset Description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
- [Discussion Forum](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [timm Models](https://github.com/huggingface/pytorch-image-models)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 Notes

- **Kaggle Environment**: This notebook is designed to run on Kaggle with GPU
- **Data Location**: Competition data is automatically available at `/kaggle/input/` on Kaggle
- **Runtime**: Full training takes 30-60 minutes with GPU T4 x2
- **Memory**: Requires ~16GB GPU memory (available on Kaggle)

## ⚠️ Troubleshooting

### Issue: "Data not found"
**Solution:** Make sure you added the competition data in Kaggle (+ Add Input)

### Issue: "CUDA out of memory"
**Solution:** Reduce `BATCH_SIZE` to 8 or 4 in config

### Issue: Loss is NaN
**Solution:** This is a known issue with DICOM loading. Use a public notebook with proper preprocessing

### Issue: "No module named 'timm'"
**Solution:** Run the first cell to install dependencies

## 🏆 Competition Strategy

**For Beginners:**
1. Start with this baseline notebook
2. Understand the complete pipeline
3. Fork high-scoring public notebooks
4. Learn from experienced competitors

**For Better Results:**
1. Use properly preprocessed DICOM data
2. Implement 5-fold cross-validation
3. Try larger models (EfficientNet-B2, ResNet50)
4. Ensemble multiple models
5. Use multi-modal learning (images + reports)

## 📧 Contact

**Author:** Bhuvaneshwari  
**GitHub:** [@Bhuvaneshwari244](https://github.com/Bhuvaneshwari244)  
**Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Radiological Society of North America (RSNA) for hosting the competition
- Kaggle for providing free GPU compute
- PyTorch and timm communities for excellent tools
- All contributors and community members

---

**⭐ If you find this helpful, please star the repository!**

**🔗 Competition Link:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
