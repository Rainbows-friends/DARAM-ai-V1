import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace
import pyopencl as cl

class FaceRecog:
    def __init__(self):
        self.embeddings_file = 'face_embeddings.pkl'
        self.known_faces_dir = r'Y:\Faceon_Project\known_faces'
        self.load_known_faces()
        self.setup_opencl()
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

    def setup_opencl(self):
        platforms = cl.get_platforms()
        self.device = platforms[0].get_devices(cl.device_type.GPU)[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        # OpenCL program to calculate Euclidean distances
        self.program = cl.Program(self.context, """
        __kernel void calculate_distances(__global const float* known_encodings, __global const float* face_encoding, __global float* distances, int num_encodings, int encoding_length) {
            int i = get_global_id(0);
            if (i < num_encodings) {
                float distance = 0.0;
                for (int j = 0; j < encoding_length; j++) {
                    float diff = known_encodings[i * encoding_length + j] - face_encoding[j];
                    distance += diff * diff;
                }
                distances[i] = sqrt(distance);
            }
        }
        """).build()

    def preprocess_input(self, frame):
        image = cv2.resize(frame, (224, 224))
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        image = image.reshape((1, 3, 224, 224))
        return image

    def process_face(self, face, frame):
        try:
            x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
            face_img = frame[y:y+h, x:x+w]
            preprocessed_frame = self.preprocess_input(face_img)

            # Get face embedding
            face_embedding = np.array(DeepFace.represent(face_img, model_name="Facenet")[0]["embedding"], dtype=np.float32)

            # Prepare OpenCL buffers
            known_encodings = np.array(self.known_face_encodings, dtype=np.float32)
            num_encodings = known_encodings.shape[0]
            encoding_length = known_encodings.shape[1]

            known_encodings_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=known_encodings)
            face_encoding_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=face_embedding)
            distances_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, known_encodings.nbytes)

            # Run OpenCL kernel
            self.program.calculate_distances(self.queue, (num_encodings,), None, known_encodings_buf, face_encoding_buf, distances_buf, np.int32(num_encodings), np.int32(encoding_length))

            # Get the result
            distances = np.empty(num_encodings, dtype=np.float32)
            cl.enqueue_copy(self.queue, distances, distances_buf).wait()

            name = "Unknown"
            if num_encodings > 0:
                best_match_index = np.argmin(distances)
                print(f"Best match distance: {distances[best_match_index]}")
                if distances[best_match_index] < 0.6:
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
            faces = DeepFace.extract_faces(frame_rgb, detector_backend='opencv', enforce_detection=False, align=False)
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
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fr = FaceRecog()
    fr.run()
