import os
import cv2
import numpy as np
from tqdm import tqdm

def load_data(base_path, img_size=50, folders_limit=10):
    folders = sorted(os.listdir(base_path))[:folders_limit]
    X, y = [], []

    print("Loading images...")
    for folder in folders:
        for label in ['0', '1']:  # 0 = benign, 1 = malignant
            path = os.path.join(base_path, folder, label)
            if not os.path.exists(path):
                continue
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (img_size, img_size))
                    X.append(img)
                    y.append(int(label))
                except:
                    pass

    X = np.array(X) / 255.0
    y = np.array(y)
    print(f"✅ Loaded {len(X)} images.")
    return X, y
