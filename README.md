# Brain Tumor Classification using CNN

## Overview

This repository contains a deep learning-based approach to classify brain tumors into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

A Convolutional Neural Network (CNN) is implemented using PyTorch to train and evaluate the model. The model is trained on MRI images of brain tumors using a deep learning approach. It utilizes a custom CNN architecture inspired by AlexNet, optimized with Adam optimizer and cross-entropy loss.The final trained model achieves an **F2-score of 0.9745** on the test dataset.

## Dataset

The dataset is organized into the following directories:

```
Data/
│── Train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   ├── pituitary/
│── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   ├── pituitary/
│── Validation/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   ├── pituitary/
```

- **Train**: Used for training the model.
- **Testing**: Used for final evaluation.
- **Validation**: Used for hyperparameter tuning.

## Model Architecture

The implemented CNN model consists of the following layers:

1. **Feature Extraction Layers**
   - Convolutional Layers
   - ReLU Activation
   - Max Pooling Layers
2. **Fully Connected Layers**
   - Dropout for regularization
   - Linear Layers
   - Softmax Output

## Training

### Hyperparameters

- **Batch Size**: 32
- **Learning Rate**: 0.0003
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 50

### Data Pre-Processing

- Grayscale conversion
- Resizing to (227, 227)
- Normalisation
  
### Data Augmentation

- Random Rotation
- Horizontal Flips
- Random Affine Transformation

## Performance Metrics

The model is evaluated using:

- **F2-score** (Prioritizing recall over precision as false negatives are critical and missing a tumor can have severe consequences for patient care. On the other hand false positives are less harmful because a false alarm might lead to extra tests but is safer than missing a tumor.)
- **Loss curve analysis**
- **Visualization of Predictions**

### Final Test Results

- **Test Loss**: Computed on the test set
- **Test F2-score**: **0.9745**

## Model Quantization

The trained model is dynamically quantized to reduce size and improve inference speed:

```python
model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(model_quantized.state_dict(), "model_quantized.pth")
```

## Visualization

Training performance is visualized using:

- **Accuracy vs. Epochs Plot**
- **Loss vs. Epochs Plot**
- **Predictions on Test Samples**

## Model Saving and Loading
The trained model is saved as `model_quantized.pth`. To load the model for inference:

```python
import torch
model = torch.load("model_quantized.pth")
model.eval()
```


## Acknowledgments

- The dataset is sourced from Kaggle: [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
- PyTorch framework for deep learning implementation inspired by AlexNet.
