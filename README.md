# CAPTCHA Breaker – End-to-End Multi-Character CNN
Deep learning system for breaking 4-character text CAPTCHAs using convolutional neural networks.
This repository implements two production-style pipelines:
1. Segmentation + CNN classifier (OpenCV + CNN)
2. End-to-end multi-character CNN (PyTorch, no segmentation)
The second approach removes rule-based contour detection and directly predicts all characters from the full image.

## End-to-End Model

**Input:** grayscale CAPTCHA image `(1×24×72)`  
**Output:** full 4-character string  

The model predicts:
````
(batch_size, 4, n_classes)
````
A prediction is correct only if **all 4 characters match**.

## Architecture

- 3 Convolution blocks (Conv → BatchNorm → ReLU → MaxPool)
- Zone-wise convolution layer
- 1×1 convolution for channel reduction
- Reshape → 4 character positions
Key idea:
Use convolutional structure instead of flattening, preserving spatial locality for character positions.

## Engineering Highlights

### Custom Multi-Character Loss

```python
cross_entropy(pred_logits.flatten(0,1), labels.flatten())
```
- Computes cross entropy across all character positions
- Enables joint optimization of full CAPTCHA prediction

### Data Augmentation

Using `torchvision.transforms.RandomAffine`:
- Rotation (±5°)
- Translation (10%)
- Scaling (0.9–1.1)
- Shearing (±5°)
- Pixel rescaling + inversion
Improves robustness and reduces overfitting.

## Performance

- Training Accuracy: ~95%+
- Validation Accuracy: ~90%+
- **Test Accuracy: 93.3%**
- 1064 / 1140 CAPTCHAs fully recognized

Model trained with:

- AdamW optimizer
- Batch normalization
- 200 epochs
- On-the-fly augmentation

## Baseline: Segmentation + CNN

Earlier version uses:
- OpenCV contour detection
- Character-level CNN classifier
- TensorFlow & PyTorch implementations
Achieves ~95% character-level accuracy.
