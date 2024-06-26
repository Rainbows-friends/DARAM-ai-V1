import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace

class FaceRecog:
    def __init__(self):
        self.embeddings_file = 'face_embeddings.pkl'
        self.load_known_faces()
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def load_known_faces(self):
        try:
            with open(self.embeddings_file, 'rb') as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            print(f"Loaded {len(self.known_face_encodings)} known faces.")
        except Exception as e:
            print(f"Error loading known faces: {e}")
            self.known_face_encodings = []
            self.known_face_names = []

    def preprocess_input(self, frame):
        image = cv2.resize(frame, (224, 224))  # assuming input size is 224x224
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def process_face(self, face, frame):
        try:
            x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
            face_img = frame[y:y+h, x:x+w]
            face_embedding = np.array(DeepFace.represent(face_img, model_name="Facenet")[0]["embedding"])
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    fr = FaceRecog()
    fr.run()