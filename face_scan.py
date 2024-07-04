import pickle

import cv2
import numpy as np
from deepface import DeepFace


class FaceRecog:
    def __init__(self):
        self.initial_embeddings_file = 'initial_face_embeddings.pkl'
        self.augmented_embeddings_file = 'augmented_face_embeddings.pkl'
        self.other_images_dir = 'known_faces/Other'
        self.load_known_faces()
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def process_face(self, face, frame):
        try:
            x, y, w, h = face
            face_img = frame[y:y + h, x:x + w]
            face_embedding = np.array(
                DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"])
            name = "Unknown"
            if self.known_face_encodings:
                known_encodings = np.array(self.known_face_encodings)
                distances = np.linalg.norm(known_encodings - face_embedding, axis=1)
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 5.0:
                    name = self.known_face_names[best_match_index]
            color = (0, 0, 255) if name == "Unknown" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"Error processing face: {e}")

    def get_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            return None, None
        try:
            faces = self.detect_faces(frame)
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecog()
    fr.run()
