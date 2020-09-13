from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import json
import base64
from PIL import Image
import requests
import os
from tqdm import tqdm

filter_data = {
    "detection": {'path': 'http://detection:5001/get_models'}
}

def detection_speed_test(url, images):
    for image in tqdm(images):
        _, buffer = cv2.imencode('.jpeg', image)
        d = 'data:image/png;base64,' + base64.b64encode(buffer).decode("utf-8")
        json={'frame': d, 'model_name': "RetinaNet"}
        response = requests.post(url=url, json=json)
        image_data = response.json()['image'].replace('data:image/png;base64,', "")
        byte_image = bytearray(base64.b64decode(image_data))
        img_input = cv2.imdecode(np.asarray(byte_image), cv2.IMREAD_COLOR)

def empty_speed_test(url, images):
    for image in tqdm(images):
        json={}
        response = requests.post(url=url, json=json)

images = []
dirpath = 'val2017'
filenames = os.listdir(dirpath)
for filename in filenames:
    image = cv2.imread(os.path.join(dirpath, filename), cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    images.append(resized_image)

url=filter_data['detection']['path'].replace('get_models', 'empty_upload')
empty_speed_test(url, images)

