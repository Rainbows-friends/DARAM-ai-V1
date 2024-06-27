import os
import requests
from bs4 import BeautifulSoup
import cv2
import dlib
import numpy as np

# 파일 경로 설정
SHAPE_PREDICTOR_PATH = r'Y:\Faceon_Project\dataset\shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = r'Y:\Faceon_Project\dataset\dlib_face_recognition_resnet_model_v1.dat'

# 이미지 저장 폴더
IMAGE_FOLDER = r'Y:\Faceon_Project\collected_images'
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# dlib의 얼굴 검출기 및 얼굴 임베딩 모델 로드
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

def download_images(query, max_images=10):
    url = f'https://www.google.com/search?q={query}&tbm=isch'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    img_urls = []
    for img in img_tags:
        if len(img_urls) >= max_images:
            break
        try:
            img_url = img['src']
            if img_url.startswith('http'):
                img_urls.append(img_url)
        except KeyError:
            continue

    for i, img_url in enumerate(img_urls):
        img_data = requests.get(img_url).content
        with open(os.path.join(IMAGE_FOLDER, f'{query}_{i}.jpg'), 'wb') as handler:
            handler.write(img_data)
    return [os.path.join(IMAGE_FOLDER, f'{query}_{i}.jpg') for i in range(len(img_urls))]

def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_rgb, 1)
    if len(dets) == 0:
        return None
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img_rgb, detection))
    face_descriptors = facerec.compute_face_descriptor(img_rgb, faces[0])
    return np.array(face_descriptors)

def main():
    query = "person"
    max_images = 10
    image_paths = download_images(query, max_images)

    embeddings = []
    for image_path in image_paths:
        embedding = get_face_embeddings(image_path)
        if embedding is not None:
            embeddings.append(embedding)

    print(f'Collected {len(embeddings)} face embeddings.')

if __name__ == "__main__":
    main()
