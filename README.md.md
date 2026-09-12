# Face Forensics: Real vs AI-Generated Face Classification

A classical computer vision project that detects AI-generated (StyleGAN) faces
using **handcrafted forensic features** — no deep learning, no CNNs.

## Project Overview

Modern GANs like StyleGAN produce faces that look photorealistic to humans,
but leave subtle statistical fingerprints:

- **Color distribution anomalies** (baseline)
- **Frequency-domain artifacts** (from upsampling layers)
- **Sensor-noise residual inconsistencies** (from the generation process)

This project tests whether these artifacts can be detected using **classical
feature engineering + traditional ML classifiers**.

## Dataset

- **Source:** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle)
- **Real images:** Flickr (Flickr-Faces-HQ)
- **Fake images:** StyleGAN-generated
- **Subset used:** 2,000 real + 2,000 fake (4,000 total)
- **Preprocessing:** Resized to 256×256, RGB for color features, grayscale for frequency features

## Methodology

### Baseline Features (96-D)
32-bin color histograms per RGB channel, concatenated into a 96-dimensional vector.

### Enhanced Features (7-D)
1. **FFT log-magnitude statistics** (3 features)
   - Mean of log-magnitude spectrum
   - Standard deviation of log-magnitude spectrum
   - High-to-low radial energy ratio (boundary at r = 64 px)
2. **Gaussian blur noise residual statistics** (4 features)
   - Mean, standard deviation, skewness, kurtosis

### Classifiers
- **Support Vector Machine** (RBF kernel, C=1.0, with StandardScaler)
- **Random Forest** (150 trees, max depth 8)

### Evaluation
- Stratified 70/30 train-test split
- Metrics: Accuracy, F1-score
- Confusion matrices for final models
- JPEG compression shortcut diagnostic (qualities 95, 75, 50)

## Results

> **To reproduce:** see [Running on Kaggle](#running-on-kaggle) below.

### Feature Comparison Table

| Feature Set | Model | Accuracy | F1-Score |
|---|---|---|---|
| Baseline (Color Hist, 96-D) | SVM (RBF) | _TBD_ | _TBD_ |
| Baseline (Color Hist, 96-D) | Random Forest | _TBD_ | _TBD_ |
| Enhanced (FFT + Residual, 7-D) | SVM (RBF) | _TBD_ | _TBD_ |
| Enhanced (FFT + Residual, 7-D) | Random Forest | _TBD_ | _TBD_ |

### Compression Shortcut Diagnostic

| Model | JPEG Quality | Accuracy | Delta vs. Original | F1 |
|---|---|---|---|---|
| SVM (RBF) | 95 | _TBD_ | _TBD_ | _TBD_ |
| SVM (RBF) | 75 | _TBD_ | _TBD_ | _TBD_ |
| SVM (RBF) | 50 | _TBD_ | _TBD_ | _TBD_ |
| Random Forest | 95 | _TBD_ | _TBD_ | _TBD_ |
| Random Forest | 75 | _TBD_ | _TBD_ | _TBD_ |
| Random Forest | 50 | _TBD_ | _TBD_ | _TBD_ |

### Interpretation

- If enhanced features outperform baseline → frequency/noise artifacts carry more signal than raw color.
- If performance collapses under re-compression → model was exploiting JPEG artifacts (shortcut learning).
- If performance holds → features capture generation-intrinsic artifacts.

## Repository Structure
