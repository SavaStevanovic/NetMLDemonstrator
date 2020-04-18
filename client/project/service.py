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

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/frame_upload', methods=['GET', 'POST'])
def frame_upload():
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imwrite('samples/image.png',img)
    
    retval, buffer = cv2.imencode('.jpeg', img)
    data = {'image':base64.b64encode(buffer).decode("utf-8") }
    # encode response using jsonpickle
    # response_pickled = json.dumps(data, ensure_ascii=False)

    # return Response(response=data, status=200, mimetype="application/json")
    return data

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False, port="5001")