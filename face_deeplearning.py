import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import pickle
import os
from deepface import DeepFace

class FaceRecog:
    def __init__(self):
        self.initial_embeddings_file = 'initial_face_embeddings.pkl'
        self.augmented_embeddings_file = 'augmented_face_embeddings.pkl'
        self.load_known_faces()
        self.update_embeddings()
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest")

    def load_known_faces(self):
        try:
            with open(self.initial_embeddings_file, 'rb') as f:
                self.initial_face_encodings, self.initial_face_names = pickle.load(f)
            print(f"Loaded {len(self.initial_face_encodings)} initial faces.")
        except Exception as e:
            print(f"Error loading initial faces: {e}")
            self.initial_face_encodings = []
            self.initial_face_names = []

        try:
            with open(self.augmented_embeddings_file, 'rb') as f:
                self.augmented_face_encodings, self.augmented_face_names = pickle.load(f)
            print(f"Loaded {len(self.augmented_face_encodings)} augmented faces.")
        except Exception as e:
            print(f"Error loading augmented faces: {e}")
            self.augmented_face_encodings = []
            self.augmented_face_names = []

        self.known_face_encodings = self.initial_face_encodings + self.augmented_face_encodings
        self.known_face_names = self.initial_face_names + self.augmented_face_names

    def update_embeddings(self):
        try:
            new_face_encodings = []
            new_face_names = []
            for filename in os.listdir(self.known_faces_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(self.known_faces_dir, filename)
                    face = DeepFace.extract_faces(img_path=img_path, enforce_detection=False, align=False)
                    if len(face) > 0:
                        face_encoding = np.array(DeepFace.represent(face[0]["face"], model_name="Facenet")[0]["embedding"])
                        new_face_encodings.append(face_encoding)
                        new_face_names.append(os.path.splitext(filename)[0])

            combined_encodings = self.initial_face_encodings.copy()
            combined_names = self.initial_face_names.copy()

            for new_encoding, new_name in zip(new_face_encodings, new_face_names):
                if new_name in combined_names:
                    idx = combined_names.index(new_name)
                    combined_encodings[idx] = np.mean([combined_encodings[idx], new_encoding], axis=0)
                else:
                    combined_encodings.append(new_encoding)
                    combined_names.append(new_name)

            with open(self.augmented_embeddings_file, 'wb') as f:
                pickle.dump((combined_encodings, combined_names), f)
            print(f"Updated embeddings for {len(new_face_encodings)} faces. Total faces: {len(combined_encodings)}.")
            self.initial_face_encodings, self.initial_face_names = combined_encodings, combined_names
        except Exception as e:
            print(f"Error updating embeddings: {e}")

    def preprocess_input(self, frame):
        image = cv2.resize(frame, (224, 224))  # assuming input size is 224x224
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def augment_image(self, image):
        img_array = np.expand_dims(image, axis=0)
        augmented_images = [image]
        for batch in self.datagen.flow(img_array, batch_size=1):
            augmented_images.append(batch[0].astype(np.uint8))
            if len(augmented_images) >= 5:  # Generate 5 augmented images
                break
        return augmented_images

    def process_face(self, face, frame):
        try:
            x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
            face_img = frame[y:y+h, x:x+w]
            augmented_images = self.augment_image(face_img)
            print(f"Augmented {len(augmented_images)} images for face at ({x}, {y}, {w}, {h})")  # 증강 시 메시지 출력
            face_embeddings = []
            for img in augmented_images:
                face_embedding = np.array(DeepFace.represent(img, model_name="Facenet")[0]["embedding"])
                face_embeddings.append(face_embedding)
            face_embedding = np.mean(face_embeddings, axis=0)
            name = "Unknown"
            if self.known_face_encodings:
                known_encodings = np.array(self.known_face_encodings)
                distances = np.linalg.norm(known_encodings - face_embedding, axis=1)
                best_match_index = np.argmin(distances)
                print(f"Best match distance: {distances[best_match_index]}")  # 디버그 정보 출력
                if distances[best_match_index] < 0.6:  # Threshold for recognizing as known face
                    name = self.known_face_names[best_match_index]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing face: {e}")

    def get_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            return None, None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False, align=False)
            for face in faces:
                self.process_face(face, frame)
        except Exception as e:
            print(f"Error extracting faces: {e}")
            cv2.putText(frame, "No match found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame, cv2.imencode('.jpg', frame)[1].tobytes()

    def run(self):
        while True:
            frame, jpg_bytes = self.get_frame()
            if frame is not None:
                cv2.imshow("Video", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('a'):  # 'a' 키를 누르면 증강 수행
                self.update_embeddings()
                print("Augmentation performed.")
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    fr = FaceRecog()
    fr.run()
