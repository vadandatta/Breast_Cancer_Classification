import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import random

# Ensure the results folder exists
os.makedirs("results", exist_ok=True)

# Load test data
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Load the trained CancerNet model
model = load_model("models/cancernet_model.h5")

# Predict on test set
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype("int32").flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels)
labels = ["Benign", "Malignant"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved at results/confusion_matrix.png")

# Classification Report
report = classification_report(y_test, y_pred_labels, target_names=labels)
with open("results/classification_report.txt", "w") as f:
    f.write(report)
print("✅ Classification report saved at results/classification_report.txt")

# Visualize sample predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    idx = random.randint(0, len(X_test) - 1)
    image = X_test[idx]
    true_label = "Malignant" if y_test[idx] == 1 else "Benign"
    pred_label = "Malignant" if y_pred_labels[idx] == 1 else "Benign"

    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig("results/example_predictions.png")
plt.close()
print("✅ Sample prediction grid saved at results/example_predictions.png")
