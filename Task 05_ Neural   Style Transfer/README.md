# Neural Style Transfer (NST) with VGG19

This repository implements **Neural Style Transfer (NST)** in TensorFlow/Keras, based on the landmark paper by Gatys et al. (2015). NST is an optimization technique that blends two images—a **content image** and a **style image**—to synthesize a new image that retains the structural content of the former while adopting the artistic texture and visual motifs of the latter.

---
## Input: <br><img width="554" height="345" alt="image" src="https://github.com/user-attachments/assets/ad14fb6f-1f12-4fb6-b92c-8f8b896a2d2c" />

## Style: <br> <img width="430" height="435" alt="image" src="https://github.com/user-attachments/assets/eb58758f-154c-46f6-8d6b-a8477100e438" />

## Output: <br> <img width="554" height="345" alt="image" src="https://github.com/user-attachments/assets/adc07041-dca4-4440-9f07-15610e294bb1" />

---

## 📌 Project Overview

Instead of training a model to predict outputs, NST optimizes pixel values of a target image directly using a pre-trained feature extractor:

* **Content Image ($x_c$):** Provides the macro-structure, shapes, and spatial layout.
* **Style Image ($x_s$):** Provides color palettes, textures, and local visual patterns.
* **Generated Target ($x$):** Initialized from the content image and iteratively modified via gradient descent to minimize a composite loss function.

---

## 🏗️ Architecture & Feature Extraction

The pipeline uses a pre-trained **VGG19** backbone (trained on ImageNet) with frozen parameters (`trainable = False`). Intermediate layer activations are extracted to evaluate content and style representations.

### Selected Layers

* **Content Representation:**
* `block5_conv2` — Captures high-level semantic layout without fine pixel detail.


* **Style Representation:**
* `block1_conv1`
* `block2_conv1`
* `block3_conv1`
* `block4_conv1`
* `block5_conv1` — Captures multi-scale textures, from low-level strokes to complex visual patterns.



---

## 📐 Loss Formulations

The total optimization objective combines three loss components:

$$\mathcal{L}_{\text{total}} = w_c \cdot \mathcal{L}_{\text{content}} + w_s \cdot \mathcal{L}_{\text{style}} + w_{\text{tv}} \cdot \mathcal{L}_{\text{TV}}$$

### 1. Content Loss

Measures the Mean Squared Error (MSE) between feature maps of the generated image ($F$) and content image ($P$) at the selected content layer:

$$\mathcal{L}_{\text{content}} = \frac{1}{2} \sum_{i, j} (F_{ij} - P_{ij})^2$$

### 2. Style Loss

Style features are captured using **Gram Matrices** ($G$), which compute feature correlations across channels within a layer:

$$G_{cd}^l = \frac{1}{N_l M_l} \sum_{k} F_{ik}^l F_{jk}^l$$

The style loss is the mean squared difference between the Gram matrices of the generated image ($G$) and style image ($A$) across all style layers:

$$\mathcal{L}_{\text{style}} = \sum_{l} w_l \cdot \frac{1}{4 N_l^2 M_l^2} \sum_{i, j} (G_{ij}^l - A_{ij}^l)^2$$

### 3. Total Variation (TV) Loss

Acts as a regularizer to reduce high-frequency noise and spatial artifacts, encouraging smooth visual outputs.

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| **Max Image Dimension** | `512 px` | Maintains aspect ratio while scaling long edge |
| **Content Weight ($w_c$)** | `1e4` | Scaling factor for content loss |
| **Style Weight ($w_s$)** | `1e2` | Scaling factor for style loss |
| **TV Weight ($w_{\text{tv}}$)** | `30` | Scaling factor for total variation regularization |
| **Optimizer** | Adam | $\text{learning\_rate} = 0.02, \beta_1 = 0.99, \epsilon = 0.1$ |
| **Optimization Iterations** | 10 Epochs (100 steps/epoch) | Total 1,000 steps |

---

## 📁 Preprocessing & Workflow

1. **Image Loading:** Images are loaded, normalized to $[0, 1]$, and resized (preserving aspect ratio with a max dimension of 512 px).
2. **Feature Extraction Class:** `StyleContentModel` wraps VGG19, scales inputs to $[0, 255]$, applies standard VGG preprocessing, and returns layer activation maps.
3. **Gradient Descent Loop:** Updates pixel values of `tf.Variable(content_image)` via `tf.GradientTape()` and clips pixel values within $[0.0, 1.0]$.

---

## 🚀 Getting Started

### Installation

Ensure required libraries are installed:

```bash
pip install tensorflow matplotlib numpy pillow

```

### Usage

```python
import os
import tensorflow as tf

# Define image paths
content_path = '/content/anime-version.jpg'
style_path = '/content/van2.jpg'

# Run style transfer pipeline...
# The final result will be saved automatically as:
# 'final_generated_image.png'

```

---

## 📊 Output Artifacts

* `final_generated_image.png` — Final output image produced after optimization.
