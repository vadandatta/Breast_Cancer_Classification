from sklearn.model_selection import train_test_split
from cancer_net_model import build_cancernet
from utils import load_data
import os

# Dataset path
base_path = 'breast-histopathology-images'
X, y = load_data(base_path)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = build_cancernet()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/cancernet.h5")
