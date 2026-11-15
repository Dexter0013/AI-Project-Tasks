# Generative-AI-GA Projects
 
Welcome to the central repository containing hands-on implementations across **Natural Language Processing (NLP)**, **Generative AI**, **Computer Vision**, and **Neural Style Transfer**.

---

## 📁 Task Directory & Readme Links

Each project is self-contained within its respective directory. Click on any link below to navigate directly to the detailed `README.md` file for that task:

| Task | Module | Tech Stack 
| --- | --- | --- |
| **Task 01** | Text Generation with GPT-2 | Transformers, PyTorch/TensorFlow 
| **Task 02** | Image Generation with Pre-trained Models | Stable Diffusion, Diffusers 
| **Task 03** | Predictive Text Generation | Markov Chains / Statistical NLP 
| **Task 04** | Image-to-Image Translation (Pix2Pix cGAN) | TensorFlow, Keras
| **Task 05** | Neural Style Transfer (NST) | TensorFlow, Keras, VGG19
---

## 📂 Complete File Structure

```text
.
├── 📁 Task 01: Text Generation with GPT.../
│   ├── 📄 Fine_Tune_GPT2_Model_using_t...
│   ├── 📄 test102.txt
│   ├── 📄 train100.txt
│   └── 📖 README.md
│
├── 📁 Task 02: Image Generation with P.../
│   ├── 📄 Generate_Images_with_Stable_D...
│   └── 📖 README.md 
│
├── 📁 Task 03: Text Generation with Ma.../
│   ├── 📄 Predective_Text_Generation_wit...
│   ├── 📄 1342-0.txt
│   ├── 📄 84-0.txt
│   └── 📖 README.md
│
├── 📁 Task 04: Image-to-Image Translat.../
│   ├── 📄 Image_to_Image_Translation_wit...
│   └── 📖 README.md 
│
├── 📁 Task 05: Neural Style Transfer/
│   ├── 📓 Neural_Style_Transfer.ipynb
│   └── 📖 README.md
│
├── 📄 LICENSE
└── 📖 README.md  (Main Parent README)

```

## 🛠️ Environment Setup

To run the implementations across all tasks, clone this repository and install the standard dependency bundle:

```bash
# Clone the repository
gh repo clone Dexter0013/AI-Project-Tasks
cd AI-Project-Tasks

# Install required dependencies
pip install tensorflow torch transformers diffusers matplotlib numpy pillow datasets pydot graphviz
