# malaria-classification
# 🦠 Malaria Cell Classification with VGG19

This project uses transfer learning with the VGG19 model to classify cell images as parasitized or uninfected.

---

## 📁 Dataset

⚠️ Dataset is not included in the repository.

Place it in this structure:

malaria_dataset/
├── train/
│ ├── Parasitized/
│ └── Uninfected/
├── val/
│ ├── Parasitized/
│ └── Uninfected/
└── test/
├── Parasitized/
└── Uninfected/

Download: [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/publication/pub9932)

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/akhilr13/malaria-classification.git
cd malaria-classification

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
