import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import xgboost as xgb
import logging

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Error reading image {filepath}. Skipping this file.")
            return None
        img = cv2.resize(img, (64, 64))
        return img.flatten()
    except Exception as e:
        logging.error(f"Error processing image {filepath}: {e}")
        return None

def load_data(positive_dir, negative_dir):
    data = []
    labels = []

    if not os.path.exists(negative_dir):
        os.makedirs(negative_dir)
        logging.info(f"Created {negative_dir} directory. Please add non-face images for training.")

    logging.info("Loading positive images...")
    positive_files = [os.path.join(positive_dir, filename) for filename in os.listdir(positive_dir) if filename.endswith(".jpg") or filename.endswith(".png")]
    positive_images = Parallel(n_jobs=-1)(delayed(load_image)(file) for file in tqdm(positive_files, desc="Positive Images"))
    positive_images = [img for img in positive_images if img is not None]
    data.extend(positive_images)
    labels.extend([1] * len(positive_images))

    logging.info("Loading negative images...")
    negative_files = [os.path.join(negative_dir, filename) for filename in os.listdir(negative_dir) if filename.endswith(".jpg") or filename.endswith(".png")]
    negative_images = Parallel(n_jobs=-1)(delayed(load_image)(file) for file in tqdm(negative_files, desc="Negative Images"))
    negative_images = [img for img in negative_images if img is not None]
    data.extend(negative_images)
    labels.extend([0] * len(negative_images))

    return np.array(data), np.array(labels)

def train_face_detector(data, labels):
    logging.info(f"Class distribution: {np.bincount(labels)}")
    if len(np.unique(labels)) < 2:
        raise ValueError("The number of classes has to be greater than one.")
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    logging.info("Training model...")
    clf = xgb.XGBClassifier(use_label_encoder=False, n_jobs=-1, tree_method='gpu_hist' if tf.config.list_physical_devices('GPU') else 'hist')
    clf.fit(X_train, y_train)
    logging.info("Predicting test data...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    return clf

if __name__ == "__main__":
    positive_dir = r'C:\Faceon_Project\DTFO_Taeeun\known_faces\Other'
    negative_dir = r'C:\Faceon_Project\DTFO_Taeeun\non_faces'
    data, labels = load_data(positive_dir, negative_dir)
    logging.info("Training face detector...")
    model = train_face_detector(data, labels)
    joblib.dump(model, r'C:\Faceon_Project\DTFO_Taeeun\face_detector.pkl')
    logging.info("Model training complete and saved as 'face_detector.pkl'")
