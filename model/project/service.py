from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import json
import base64
import torch
from torch2trt import torch2trt, TRTModule
from torchvision.models.alexnet import alexnet
from torch2trt import TRTModule
from visualization import output_transform
from visualization import apply_output
from PIL import Image
from data_loader import augmentation
import imutils

app = Flask(__name__)


transfor = augmentation.OutputTransform()
camera_models = {}

model_paths = {
        "YoloV2" : {'path': 'checkpoints/YoloV2/64/0,5-1,0-2,0/Coco_checkpoints_final.pth'},
        "Yolo" : {'path': 'checkpoints/YoloNet/512/0,5-1,0-2,0/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        "RetinaNet" : {'path': 'checkpoints/RetinaNet/512/0,5-1,0-2,0/FeaturePyramidBackbone/512/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        "FPN" : {'path': 'checkpoints/FeaturePyramidNet/2048/0,5-1,0-2,0/FeaturePyramidBackbone/2048/Coco_checkpoints_final.pth'},
    }


@app.route('/get_models', methods=['GET', 'POST'])
def get_models():
    return jsonify(list(model_paths.keys()))


@app.route('/frame_upload', methods=['GET', 'POST'])
def frame_upload():
    data = request.get_json()
    image_data = data['frame'].replace('data:image/png;base64,', "")
    byte_image = bytearray(base64.b64decode(image_data))
    img_input = cv2.imdecode(np.asarray(byte_image), cv2.IMREAD_COLOR)
    img = imutils.resize(img_input, height=256)
    model_key = data['model_name']
    if model_key not in camera_models:
        if model_key not in model_paths:
            raise Exception("Model {} not found.".format(model_key))
        model_path = model_paths[model_key]['path']
        model = torch.load(model_path).eval().cuda()
        model.target_to_box_transform = output_transform.TargetTransformToBoxes(prior_box_sizes=model.prior_box_sizes,
                                                                                classes=model.classes,
                                                                                ratios=model.ratios,
                                                                                strides=model.strides)
        model.padder = augmentation.PaddTransform(pad_size=2**model.depth)
        camera_models[model_key] = model
    model = camera_models[model_key]
    padded_img, _ = model.padder(Image.fromarray(img), None)
    img_tensor, _ = transfor(padded_img, None)
    img_tensor = img_tensor.unsqueeze(0).float().cuda()

    outputs = model(img_tensor)
    outs = [out.cpu().detach().numpy() for out in outputs]
    for out in outs:
        img = apply_output.apply_detections(model.target_to_box_transform, out, [
        ], Image.fromarray(img), model.classes, 0.5)

    img = cv2.resize(img, dsize=img_input.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    retval, buffer = cv2.imencode('.jpeg', img)
    data = {'image': 'data:image/png;base64,' + base64.b64encode(buffer).decode("utf-8")}

    return data

@app.route('/decode_upload', methods=['GET', 'POST'])
def decode_upload():
    data = request.get_json()
    image_data = data['frame'].replace('data:image/png;base64,', "")
    byte_image = bytearray(base64.b64decode(image_data))
    img_input = cv2.imdecode(np.asarray(byte_image), cv2.IMREAD_COLOR)
    retval, buffer = cv2.imencode('.jpeg', img_input)
    data = {'image': 'data:image/png;base64,' + base64.b64encode(buffer).decode("utf-8")}

@app.route('/empty_upload', methods=['GET', 'POST'])
def empty_upload():
    return request.get_json()

    return data
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port="5001", threaded=False)
