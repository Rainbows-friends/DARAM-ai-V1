from flask import Flask, render_template, Response
import threading
import face_scan

app = Flask(__name__)
fr = face_scan.FaceRecog()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame, jpg_bytes = fr.get_frame()
        if jpg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=fr.run, daemon=True).start()
    app.run(host='0.0.0.0', debug=True)
