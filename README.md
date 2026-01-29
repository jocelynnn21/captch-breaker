# CAPTCHA Breaker: Deep Learning Approach

A deep learning project that demonstrates how traditional text-based CAPTCHAs can be automatically solved using Convolutional Neural Networks (CNNs). This project implements the solution in both **TensorFlow/Keras** and **PyTorch** to compare the two popular deep learning frameworks.

## 🎯 Project Overview

CAPTCHAs (Completely Automated Public Turing test to tell Computers and Humans Apart) were designed to distinguish humans from bots. However, with advances in computer vision and deep learning, many traditional CAPTCHAs can now be solved programmatically. This project explores the effectiveness of CNNs in breaking simple text-based CAPTCHAs.

## 🚀 Features

- **Automated character segmentation** using OpenCV contour detection
- **CNN-based character recognition** with >95% accuracy
- **Dual implementation**: Both TensorFlow/Keras and PyTorch versions
- **End-to-end pipeline**: From raw CAPTCHA images to predicted text
- **Comprehensive preprocessing**: Image transformation, feature extraction, and label encoding

## 🏗️ Architecture

The CNN architecture consists of:
- **2 Convolutional blocks** (Conv2D + ReLU + MaxPooling)
- **Flatten layer** to convert 2D features to 1D
- **2 Fully-connected layers** for classification
- **Softmax output** for character probability distribution

```
Input (20×20×1) 
    ↓
Conv2D (20 filters, 5×5) + ReLU + MaxPool
    ↓
Conv2D (50 filters, 5×5) + ReLU + MaxPool
    ↓
Flatten (1250 features)
    ↓
Dense (500 units) + ReLU
    ↓
Dense (n_classes) + Softmax
    ↓
Output (character probabilities)
```

## 📊 Pipeline

1. **Load CAPTCHA images** from dataset
2. **Preprocess images**: Convert to grayscale, add padding
3. **Extract characters**: Use contour detection to isolate individual characters
4. **Feature extraction**: Resize characters to 20×20 pixels
5. **Train CNN model**: Learn character patterns
6. **Predict**: Run full CAPTCHAs through the pipeline
7. **Evaluate**: Measure accuracy on test set

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/captcha-breaker.git
cd captcha-breaker

# Install dependencies
pip install numpy opencv-python imutils scikit-learn matplotlib

# For TensorFlow implementation
pip install tensorflow

# For PyTorch implementation
pip install torch torchvision
```

## 📦 Dataset

The project uses a dataset of 4-character CAPTCHAs. Each image filename contains the ground truth text (e.g., `2A2X.png`).

```bash
# Extract the dataset
tar -xJf captcha-images.tar.xz
```

## 🎮 Usage

### TensorFlow/Keras Version

```bash
jupyter notebook Breaking-CAPTCHAS-TensorFlow.ipynb
```

### PyTorch Version

```bash
jupyter notebook Breaking-CAPTCHAS-Pytorch.ipynb
```

Both notebooks include:
- Data preprocessing and visualization
- Model training with validation
- Performance evaluation
- Sample predictions on test CAPTCHAs

## 📈 Results

| Metric | TensorFlow | PyTorch |
|--------|-----------|---------|
| Training Accuracy | ~98% | ~98% |
| Validation Accuracy | ~96% | ~96% |
| Test Accuracy | ~95% | ~95% |
| Training Time (10 epochs) | ~2-3 min | ~2-3 min |

*Results may vary based on hardware and random initialization*
