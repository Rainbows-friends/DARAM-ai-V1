import os
import pickle
from deepface import DeepFace

def create_face_embeddings(directory='known_faces', output_file='initial_face_embeddings.pkl'):
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir) and person_name != "Other":
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    face = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
                    if face:
                        face_embedding = face[0]["embedding"]
                        known_face_encodings.append(face_embedding)
                        known_face_names.append(person_name)
                    else:
                        print(f"No face detected in {image_path}. Skipping this image.")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    create_face_embeddings()