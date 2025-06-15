# 🦠 Malaria Cell Classification using VGG19

This project implements a deep learning-based binary classifier to detect malaria-infected blood cells using a fine-tuned **VGG19** Convolutional Neural Network (CNN). The model is trained on labeled microscopy images and achieves high classification performance.

---

## 📁 Project Structure

malaria-vgg19/
│
├── train/ # Training dataset (not included)
├── val/ # Validation dataset (not included)
├── test/ # Testing dataset (not included)
│
├── vgg19_malaria.h5 # Trained model (excluded from repo, see download link below)
├── malaria_classification.py # Main training and evaluation script
├── requirements.txt # Required Python packages
└── README.md # Project documentation


---

## ✅ Features

- 📊 Binary classification: **Parasitized** vs **Uninfected** cells
- 🧠 Transfer learning with **VGG19** pretrained on ImageNet
- 🔄 Data augmentation for robust training
- 📈 Training + validation performance tracking
- 💾 Model saving for future inference

---

## 🧪 Model Architecture

Utilizes VGG19 as a feature extractor with a custom classification head:
- Global Average Pooling
- Dense layer (128 units + ReLU)
- Dropout (0.5)
- Output layer with Sigmoid activation

---

## 🧠 Training Configuration

- **Input size**: 224x224 pixels
- **Optimizer**: Adam (learning rate = 1e-4)
- **Loss**: Binary Crossentropy
- **Epochs**: 10
- **Batch size**: 32

---

## 🚀 How to Run

> 📝 **Note**: Dataset not included due to size. You can download it from the [official NIH source](https://lhncbc.nlm.nih.gov/publication/pub9932).

### 1. Clone the Repository
```bash
git clone https://github.com/akhilr13/malaria-classification.git
cd malaria-classification



#setup Environment
pip install -r requirements.txt



#Dataset Structure
malaria_dataset/
├── train/
│   ├── Parasitized/
│   └── Uninfected/
├── val/
│   ├── Parasitized/
│   └── Uninfected/
└── test/
    ├── Parasitized/
    └── Uninfected/


# Train the Model
python malaria_classification.py
