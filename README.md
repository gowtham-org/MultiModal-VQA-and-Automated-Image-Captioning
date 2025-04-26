# Multimodal AI for Visual Question Answering (VQA) and Automated Image Captioning

This document outlines the steps taken during the project development, including the motivation, environment setup, dataset processing, model training, evaluation, and deployment strategies.


## 1. Project Overview
The goal of this project is to enhance machine understanding of images by combining visual data with advanced language models. We address two primary tasks:

- **Image Captioning**: Automatically generating descriptive captions for images.
- **Visual Question Answering (VQA)**: Answering questions related to image content.

## 2. Motivation
Recent advancements in Artificial Intelligence have led to the rise of **multimodal AI**, where models process multiple types of data such as text, images, and audio.

- **Key Technologies**:
  - Transformers (NLP revolution)
  - Vision Transformers (ViTs)
  - Multimodal models (combining vision and language)

## 3. Methodology
### 3.1 Environment Setup
- Verified GPU (CUDA) availability.
- Configured Hugging Face cache directories.

### 3.2 Installing Dependencies
- Installed libraries like `datasets`, `fsspec`, and `transformers`.

### 3.3 Dataset Loading & Preprocessing
- Loaded DAQUAR dataset via Hugging Face.
- Applied text tokenization using pretrained models.

### 3.4 Model Initialization
- Initialized vision encoders (e.g., CNNs, ViTs) and transformer-based text models.

### 3.5 Training Process
- Used `TrainingArguments` and Hugging Face `Trainer` for model training.

### 3.6 Evaluation & Metrics
- Evaluated models on validation data.
- Metrics: **Accuracy**, **F1-score**.

### 3.7 Deployment Strategy
- Prepared models for inference via Hugging Face Pipelines, TorchScript, or FastAPI.

## 4. Technologies Used
- **Pretrained Models**:
  - [BLIP (Bootstrapping Language-Image Pre-training)](https://huggingface.co/Salesforce/blip-image-captioning-base)
  - [ViLT (Vision-and-Language Transformer)](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- **Libraries**:
  - Hugging Face `transformers`
  - Hugging Face `datasets`
  - PyTorch

## 5. Model Architecture
- **Vision Encoder**: Extracts features from images.
- **Text Encoder/Decoder**: Processes and generates textual data.
- **Multimodal Fusion**: Combines image and text features.
- **Training Objectives**:
  - Classification loss for VQA.
  - Sequence generation loss for captioning.

## 6. Results
- Successfully trained a multimodal AI model capable of:
  - Generating captions for images.
  - Answering questions about image content.
- Application areas:
  - Accessibility tools.
  - Educational platforms.
  - Content management systems.

## 7. Conclusion
We developed a multimodal AI model leveraging pretrained transformers and vision encoders, effectively handling image captioning and visual question answering tasks. Feature engineering and fine-tuning enhanced the model's accuracy and applicability in real-world scenarios.

## 8. How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo-link.git
    cd your-repo-link
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Jupyter Notebook:
    ```bash
    jupyter notebook DIP_Team_3_Final_Project.ipynb
    ```

## 10. Acknowledgements
- Hugging Face for providing models and datasets.
- PyTorch for deep learning framework support.
- Microsoft Research for contributions to multimodal transformer research.
