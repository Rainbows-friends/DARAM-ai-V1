import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def train_model(embeddings_file='face_embeddings.pkl'):
    with open(embeddings_file, 'rb') as f:
        face_encodings, face_names = pickle.load(f)

    le = LabelEncoder()
    labels = le.fit_transform(face_names)
    labels = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(face_encodings, labels, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, input_dim=len(face_encodings[0]), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(le.classes_), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=8, validation_data=(np.array(X_test), np.array(y_test)))

    model.save('face_recognition_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

if __name__ == "__main__":
    train_model()
