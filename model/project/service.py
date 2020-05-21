from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import json
import base64
import torch
from torch2trt import torch2trt, TRTModule
from torchvision.models.alexnet import alexnet
from torch2trt import TRTModule
from network_output_processor import output_transform
from network_output_processor import apply_output
from PIL import Image
from data_loader import augmentation

app = Flask(__name__)

model_path = 'checkpoints/YoloNet/256/0,5-1,0-2,0/ResNetBackbone/256/3-4-6-3/SqueezeExcitationBlock/Coco_checkpoints_final.pth'
model = torch.load(model_path).eval().cuda()

feature_range = range(model.feature_start_layer, model.feature_start_layer + model.feature_count)
prior_box_sizes = [32*2**i for i in feature_range]
strides = [2**(i+1) for i in feature_range]
target_to_box_transform = output_transform.TargetTransformToBoxes(prior_box_sizes=prior_box_sizes, classes=model.classes, ratios=model.ratios, strides=strides)
padder = augmentation.PaddTransform(pad_size=2**model.depth)
transfor = augmentation.OutputTransform()
camera_models = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frame_upload', methods=['GET', 'POST'])
def frame_upload():
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    padded_img, _ = padder(Image.fromarray(img), None)
    # cv2.imwrite('samples/image.png',img)
    model_key = '_'.join(str(x) for x in list(img.shape))
    img_tensor, _ = transfor(padded_img, None)
    img_tensor = img_tensor.unsqueeze(0).float().cuda()
    if model_key not in camera_models:
        camera_models[model_key] = torch2trt(model, [img_tensor])
    outputs = camera_models[model_key](img_tensor)
    outs = [out.cpu().numpy() for out in outputs]

    pilImage = apply_output.apply_detections(target_to_box_transform, outs, [], Image.fromarray(img), model.classes)

    img = np.array(pilImage)[:img.shape[0], :img.shape[1]]
    retval, buffer = cv2.imencode('.jpeg', img)
    data = {'image':base64.b64encode(buffer).decode("utf-8") }

    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port="5001", threaded=True)