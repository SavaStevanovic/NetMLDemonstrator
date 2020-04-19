from flask import Flask, render_template, Response, jsonify, request, send_file
from camera import VideoCamera
import cv2
import numpy as np
import json
import base64

app = Flask(__name__)

video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frame_upload', methods=['GET', 'POST'])
def frame_upload():
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imwrite('samples/image.png',img)
    
    retval, buffer = cv2.imencode('.jpeg', img)
    data = {'image':base64.b64encode(buffer).decode("utf-8") }

    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port="5001", threaded=True)