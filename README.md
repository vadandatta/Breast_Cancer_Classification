
# Breast Cancer Classification using Convolutional Neural Network (CancerNet)

This project implements a deep learning model to classify breast cancer histopathology images as either benign or malignant. The model is trained on high-resolution microscopy images using a Convolutional Neural Network architecture (CancerNet) tailored for binary classification.

---

##  Objective

To assist pathologists by automating the detection of invasive ductal carcinoma (IDC) in breast tissue slides through image classification, thereby enhancing early diagnosis and treatment planning.

---

##  Dataset

- Source: Kaggle – Breast Histopathology Images
- Author: Paul Mooney
- Link: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
- Image Dimensions: 50x50 px RGB
- Classes:
  - 0 = No IDC (Benign)
  - 1 = IDC (Malignant)
- Total images: 277,524

You must download and prepare the dataset from the link above. Place the dataset in a folder named data/.

---

##  Project Structure

```

Breast-Cancer-Classification/
├── data/                              # Histopathology image patches
├── models/                            # Saved CNN model (CancerNet.h5)
├── results/
│   ├── confusion\_matrix.png           # Confusion matrix plot
│   └── example\_predictions.png        # Sample predictions
├── report/
│   └── Breast\_Cancer\_Classification\_Report.pdf
├── src/
│   ├── data\_loader.py                 # Loads and splits image dataset
│   ├── model.py                       # CancerNet CNN architecture
│   ├── train\_model.py                 # Trains CNN on training data
│   └── evaluate\_model.py              # Generates evaluation plots
├── main.py                            # Main pipeline script
├── requirements.txt
├── README.md
└── .gitignore

````

---

##  Model Architecture (CancerNet)

- Input: 50x50 RGB images
- 3× (Conv2D → BatchNorm → ReLU → MaxPooling)
- Flatten → Dense(64) → Dropout
- Output: Dense(1) with Sigmoid
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy

---

##  Training Process

- Batch size: 32
- Epochs: 10–20
- Augmentation: Horizontal flip, rotation, zoom
- Model saved as: models/CancerNet.h5

---

##  Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Visual Results:

- results/confusion_matrix.png  
- results/example_predictions.png

---

##  How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ````

2. Download the Kaggle dataset and extract into a folder named data/

3. Train the model:

   ```bash
   python src/train_model.py
   ```

4. Evaluate the model:

   ```bash
   python src/evaluate_model.py
   ```

---

## 📄 Report

A detailed report outlining methodology, results, and implications is available at:

report/Breast\_Cancer\_Classification\_Report.pdf

---

##  Author

* Vadan Datta

---

##  License

This project is for academic and educational use only. Please cite appropriately if reused.
