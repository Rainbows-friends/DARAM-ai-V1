import multiprocessing
import os

import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tqdm import tqdm

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["TF_NUM_INTEROP_THREADS"] = str(multiprocessing.cpu_count())


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)
    return images, labels


def prepare_dataset():
    base_dir = 'C:/Faceon_Project/DTFO_Taeeun/known_faces'
    classes = os.listdir(base_dir)
    images = []
    labels = []
    valid_classes = []
    class_counts = []

    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        class_images, class_labels = load_images_from_folder(class_dir, idx)
        if len(class_images) < 2:
            print(f"Class '{class_name}' has only {len(class_images)} samples. Skipping this class.")
            continue
        images.extend(class_images)
        labels.extend(class_labels)
        valid_classes.append(class_name)
        class_counts.append(len(class_images))

    negative_dir = 'C:/Faceon_Project/DTFO_Taeeun/non_faces'
    negative_images, negative_labels = load_images_from_folder(negative_dir, len(valid_classes))

    images.extend(negative_images)
    labels.extend(negative_labels)

    return np.array(images), np.array(labels), valid_classes


data, labels, class_names = prepare_dataset()
data = data.reshape(data.shape[0], 64, 64, 1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, num_classes=len(class_names) + 1)
y_test = to_categorical(y_test, num_classes=len(class_names) + 1)

model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)), MaxPooling2D((2, 2)), Dropout(0.25),

                    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.25),

                    Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.25),

                    Flatten(), Dense(128, activation='relu'), Dropout(0.5),
                    Dense(len(class_names) + 1, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

model.save('C:/Faceon_Project/DTFO_Taeeun/face_detector_cnn.h5')
