# 🎨 Image Generation with Stable Diffusion

A lightweight Python notebook pipeline to generate high-quality, cinematic images from text prompts using Hugging Face's `diffusers` library and PyTorch.

---

## ⚡ Quickstart

### 1. Installation
Install the necessary dependencies for running the Stable Diffusion pipeline:

```bash
pip install -q diffusers transformers accelerate scipy
```

### 2. Pipeline Overview
1. **Load Pre-trained Pipeline:** Downloads and initializes `runwayml/stable-diffusion-v1-5` in `float16` precision for fast inference.
2. **GPU Acceleration:** Offloads model computations to CUDA GPU memory.
3. **Prompt Execution:** Passes text prompts through the diffusion process without tracking gradients (`torch.no_grad()`).
4. **Image Rendering:** Displays the resulting generated image inline.

---

## 💻 Code Implementation

```python
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display

# 1. Load the model pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)
pipeline.to("cuda")

print(f"Stable Diffusion pipeline '{model_id}' loaded successfully.")

# 2. Define text prompt
sd_prompt = "Dog, with sunset scenery on the background, cinematic view"

# 3. Generate image
with torch.no_grad():
    image = pipeline(sd_prompt).images[0]

# 4. Display result
display(image)
print(f"Image generated for prompt: '{sd_prompt}'")
```

---

## Results

| Prompt | Output Result |
| :--- | :--- |
| `"Dog, with sunset scenery on the background, cinematic view "` | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/7e1741fe-54ed-4c47-b035-c2c0da8398a1" /> |
| `"a man sitting, with sunset scenery on the "`</n>`" background, cinematic view "` | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/f9674944-5027-4624-a9cb-0ab3150dfa03" />
  |

---
