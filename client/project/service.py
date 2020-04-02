from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import cv2

app = Flask(__name__)

video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/frame_upload', methods=['GET', 'POST'])
def frame_upload():
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False, port="5000")