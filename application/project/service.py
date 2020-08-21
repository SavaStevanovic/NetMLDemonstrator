from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import json
import base64
from PIL import Image
import requests

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

filter_data = [
    {'name':"detection", 'path':'http://detection:5001/get_models'}
]

@app.route('/get_filters', methods=['GET', 'POST'])
@cross_origin()
def get_filters():
    filters = [{'name': "Test", 'models': ['Test_good', 'Test_bad']}]

    for i, d in enumerate(filter_data):
        try:
            response = requests.get(d['path'])
            if response.status_code == 200:
                filters.append({'name':d['name'], 'models':response.json()})
        except Exception as e:
            pass

    return jsonify(filters)

@app.route('/frame_upload', methods=['GET', 'POST'])
@cross_origin()
def frame_upload():
    data = request.get_json()
    used_models = [x for x in data['config'] if 'selectedModel' in x]
    for model_config in used_models:
        model_services = [x for x in filter_data if x['name']==model_config['name']]
        if len(model_services)>0:
            model_service = model_services[0]
            r = requests.post(url = model_service['path'].replace('get_models', 'frame_upload'), json = {'frame': data['frame'], 'model_name': model_config['selectedModel']}) 
            return r.json()

    image_data = data['frame'].replace('data:image/png;base64,', "")
    byte_image = bytearray(base64.b64decode(image_data))
    frame = cv2.imdecode(np.asarray(byte_image), cv2.IMREAD_COLOR)
    ret_data = {'image':data['frame'] }

    return ret_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port="4321", threaded=False)