import pickle

with open('../face_embeddings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

print(f"Loaded {len(known_face_encodings)} known faces.")
print("First 5 names:", known_face_names[:5])
print("First 5 encodings:", known_face_encodings[:5])