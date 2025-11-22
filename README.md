# Hybrid Spatio-Temporal YOLO (HST-YOLO)
### CLAHE + Gamma + ConvLSTM Fusion for Advanced Nighttime Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

**Hybrid Spatio-Temporal YOLO** is an advanced object detection architecture specifically engineered for **extremely low-light environments**. Traditional detectors often fail when camera sensors capture underexposed, noisy, or low-contrast frames.



This model solves these challenges by combining three distinct stages into a unified architecture:
1.  **Image Enhancement:** A preprocessing pipeline using CLAHE and Gamma correction.
2.  **Temporal Modeling:** Utilizing ConvLSTM to maintain object continuity across frames.
3.  **Multi-Branch Feature Fusion:** capturing contextual information at various receptive fields.

It is particularly effective in scenarios with heavy glare, fluctuating illumination, dark shadows, and unpredictable car headlights.

---

## 🚀 Key Features

* **Adaptive Preprocessing Pipeline:** * **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Improves local contrast in dark regions without over-amplifying noise.
    * **Gamma Correction:** Modifies global luminance to recover mid-tone visibility.
* **Temporal Reasoning (ConvLSTM):** Unlike static detectors, this model uses sequence data to infer object boundaries even when frames are partially dark or blurred.
* **Multi-Branch Fusion Block:** Fuses features from parallel convolutions (1×1, 3×3, 5×5, 7×7) via concatenation and residual connections to handle variable visibility.
* **Robust Loss Function:** optimized using a combination of Detection Loss, Temporal Stability Loss, and Attention Regularization.

---

## 🏗️ Architecture
<img width="569" height="642" alt="image" src="https://github.com/user-attachments/assets/c74bc690-0e0d-4990-bf39-d15c73e5c029" />

The model pipeline processes video sequences through the following stages:
<img width="435" height="470" alt="image" src="https://github.com/user-attachments/assets/c17e597c-607b-473d-b803-229ed5e26047" />
<img width="404" height="411" alt="image" src="https://github.com/user-attachments/assets/4dd5313a-783a-4eaf-91b3-cc5150b1c271" />

```mermaid
graph TD
    Input[Input Video Sequence] --> Pre[Preprocessing Stage]
    Pre -->|CLAHE + Gamma| Backbone[YOLO-like Backbone]
    Backbone -->|Multi-scale Features| CLSTM[ConvLSTM Units]
    CLSTM -->|Temporal Features| Fusion[Multi-Branch Fusion Block]
    Fusion -->|1x1, 3x3, 5x5, 7x7 Convs| Head[Detection Head]
    Head --> Output[Final Bounding Boxes]
