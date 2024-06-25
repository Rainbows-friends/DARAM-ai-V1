import cv2
import numpy as np
from openvino.runtime import Core
from deepface import DeepFace
import pickle

class FaceRecog:
    def __init__(self):
        self.embeddings_file = 'face_embeddings.pkl'
        self.load_known_faces()
        self.setup_model()
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

    def setup_model(self):
        try:
            self.core = Core()
            self.model = self.core.read_model(model="Y:\\Faceon_Project\\dataset\\model.xml")
            config = {
                "NUM_STREAMS": "2",
                "INFERENCE_PRECISION_HINT": "f32",
                "PERFORMANCE_HINT": "THROUGHPUT",
            }
            self.compiled_model = self.core.compile_model(self.model, device_name="GPU", config=config)
            self.input_layer = self.compiled_model.input(0).any_name
            self.output_layer = self.compiled_model.output(0).any_name
            self.input_shape = self.model.input(0).partial_shape
            self.input_height, self.input_width = int(self.input_shape[1].get_length()), int(self.input_shape[2].get_length())
        except Exception as e:
            print(f"Error setting up model: {e}")

    def preprocess_input(self, frame):
        image = cv2.resize(frame, (self.input_width, self.input_height))
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.reshape((1, self.input_height, self.input_width, 3))
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
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    fr = FaceRecog()
    fr.run()