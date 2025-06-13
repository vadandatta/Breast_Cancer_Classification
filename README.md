# 🧠 Breast Cancer Classification using CNN (CancerNet)

This project focuses on building a Convolutional Neural Network (CNN)-based model named **CancerNet** to classify breast cancer histopathology images into **Benign** or **Malignant** categories.

We used the **IDC breast histopathology image dataset** from Kaggle and developed a deep learning pipeline that includes preprocessing, model training, evaluation, and visualization of results.

---

## 📁 Project Structure

├── cancer_net_model.py # CancerNet CNN model architecture
├── evaluate_model.py # Evaluation script (confusion matrix, classification report)
├── train_model.py # Training pipeline (data loading, training, saving model)
├── utils.py # Helper functions (data preprocessing, visualization)
├── requirements.txt # All required Python packages
├── README.md # You're here!
├── .gitignore # Files to be excluded from Git
├── saved_models/
│ └── cancernet_model.h5 # Trained CNN model
├── results/
│ ├── confusion_matrix.png
│ ├── classification_report.txt
│ └── example_predictions.png
└── report/
└── Breast_Cancer_Classification_Report.pdf

---

## 📊 Dataset

- **Dataset**: IDC (Invasive Ductal Carcinoma) Breast Histopathology Images  
- **Source**: [IDC Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

> ⚠️ Due to GitHub file size restrictions, the dataset is **not included** in this repository. Please download it from the Kaggle link above.

---

## 🧠 Model Architecture: CancerNet

- 3 Convolutional blocks:
  - Conv2D → BatchNorm → ReLU → MaxPooling
- Flatten + Dense layers with Dropout
- Final dense layer with `sigmoid` activation for binary classification
- Compiled using **Adam Optimizer** with **binary crossentropy** loss

---

## 🛠️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
2.Prepare Dataset
Download the dataset from Kaggle and unzip it into a directory, e.g., data/IDC_regular_ps50_idx5/

3.Train the Model
python train_model.py

4.Evaluate the Model
python evaluate_model.py

📄 Final Report
You can find the full documentation of methods, model details, evaluation, and conclusions in: report/Breast_Cancer_Classification_Report.pdf

👨‍💻 Contributors
Vadan Datta
