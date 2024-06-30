import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
import threading
import time

app = Flask(__name__)

global_frame = None
lock = threading.Lock()


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, (64, 64)))
            labels.append(label)
    return images, labels


def prepare_dataset():
    base_dir = 'C:/Faceon_Project/DTFO_Taeeun/known_faces'
    classes = os.listdir(base_dir)
    images = []
    labels = []

    for idx, class_name in enumerate(classes):
        class_images, class_labels = load_images_from_folder(os.path.join(base_dir, class_name), idx)
        images.extend(class_images)
        labels.extend(class_labels)

    negative_images, negative_labels = load_images_from_folder('C:/Faceon_Project/DTFO_Taeeun/non_faces', len(classes))

    images.extend(negative_images)
    labels.extend(negative_labels)

    return np.array(images), np.array(labels), classes


def generate_frames():
    global global_frame, lock
    camera = None
    for i in range(5):
        camera = cv2.VideoCapture(i)
        if camera.isOpened():
            print(f"Camera opened at index {i}")
            break
        camera = None

    if not camera or not camera.isOpened():
        print("Could not open any camera")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break
        else:
            # 얼굴 인식을 추가
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            with lock:
                global_frame = buffer.tobytes()
        time.sleep(0.1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    global global_frame, lock

    def generate():
        while True:
            with lock:
                if global_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
            time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    threading.Thread(target=generate_frames).start()
    app.run(debug=True, threaded=True)