# Pneumonia-MultiModel-Ensemble
Deep learning framework for automated pneumonia detection from chest X-ray images using transfer learning, data augmentation, and ensemble-based medical image classification. Built with PyTorch using ResNet50, evaluation metrics, ROC analysis, confusion matrices, and visualization for reliable AI-assisted diagnosis.

# Pneumonia-MultiModel-Ensemble

A deep learning-based medical imaging project for automated pneumonia detection from chest X-ray images using transfer learning and ensemble learning techniques.

## Overview

This project focuses on detecting pneumonia from chest X-ray images using advanced deep learning models implemented in PyTorch. The framework applies transfer learning, image preprocessing, augmentation, and performance evaluation to build a reliable AI-assisted diagnostic system.

The project includes:

- Chest X-ray preprocessing and augmentation
- Transfer learning with ResNet50
- Binary classification:
  - Normal
  - Pneumonia
- Training and validation pipelines
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC Curve
  - AUC Score
  - Confusion Matrix
- Visualization of learning curves

---

## Dataset

The notebook experiments use publicly available medical imaging datasets including:

- Pediatric Chest X-ray Pneumonia Dataset
- COVID-19 Radiography Dataset
- Chest X-ray Pneumonia Dataset

Datasets are loaded from Kaggle.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy

---

## Model Architecture

The project uses:

### Transfer Learning
- Pretrained ResNet50 backbone

### Training Features
- Image augmentation
- Fine-tuning
- Frozen feature extractor layers
- Adam optimizer
- CrossEntropy loss

---

## Image Preprocessing

- Resize to 224×224
- Normalization
- Random horizontal flip
- Random rotation
- Tensor conversion

---

## Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Classification Report
- ROC-AUC Curve
- Precision-Recall Analysis
- Confusion Matrix

---

## Results

The framework demonstrates strong performance for pneumonia classification and provides detailed visual evaluation for medical image analysis.

Generated outputs include:

- Accuracy curves
- Loss curves
- ROC curves
- Confusion matrix heatmaps

---
<img width="1800" height="1500" alt="confusion_matrix_vit" src="https://github.com/user-attachments/assets/bfad9203-4f36-4a14-ad33-42d883f4648b" />

<img width="1542" height="1407" alt="confusion_matrix" src="https://github.com/user-attachments/assets/0cac2c32-e0b2-4ec0-b4f7-bcc27f969026" />

<img width="1542" height="1407" alt="confusion_matrix_ensemble" src="https://github.com/user-attachments/assets/be318646-091d-43e3-b917-2babd525d293" />




## Project Structure

```bash
Pneumonia-MultiModel-Ensemble/
│
├── notebook.ipynb
├── models/
├── outputs/
├── plots/
├── README.md
└── requirements.txt

Author

Sajid Hussain
AI Researcher | Computer Vision & Medical Imaging
Deep Learning • Transformers • Medical AI • Explainable AI
