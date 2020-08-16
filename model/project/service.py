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

model_path = 'checkpoints/YoloV2/64/0,5-1,0-2,0/Coco_checkpoints.pth'
model = torch.load(model_path).eval().cuda()

# feature_range = range(model.feature_start_layer, model.feature_start_layer + model.feature_count)
prior_box_sizes = model.prior_box_sizes
strides = model.strides
target_to_box_transform = output_transform.TargetTransformToBoxes(prior_box_sizes=prior_box_sizes, classes=model.classes, ratios=model.ratios, strides=strides)
padder = augmentation.PaddTransform(pad_size=2**model.depth)
transfor = augmentation.OutputTransform()
camera_models = {}

model_paths  = [
        {'name':"YoloV2", 'path':'checkpoints/YoloV2/64/0,5-1,0-2,0/Coco_checkpoints_final.pth'},
        {'name':"Yolo", 'path':'checkpoints/YoloNet/512/0,5-1,0-2,0/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        {'name':"RetinaNet", 'path':'checkpoints/RetinaNet/512/0,5-1,0-2,0/FeaturePyramidBackbone/512/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        {'name':"FPN", 'path':'checkpoints/FeaturePyramidNet/2048/0,5-1,0-2,0/FeaturePyramidBackbone/2048/Coco_checkpoints_final.pth'},
    ]

@app.route('/get_models', methods=['GET', 'POST'])
def get_models():
    models = [d['name'] for d in model_paths]

    return jsonify(models)

@app.route('/frame_upload', methods=['GET', 'POST'])
def frame_upload():
    data = request.get_json()
    image_data = data['frame'].replace('data:image/png;base64,', "")
    byte_image = bytearray(base64.b64decode(image_data))
    img_input = cv2.imdecode(np.asarray(byte_image), cv2.IMREAD_COLOR)
    # img_input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = imutils.resize(img_input, height=256)
    padded_img, _ = padder(Image.fromarray(img), None)
    # cv2.imshow("Display window",img)
    # cv2.waitKey(1)
    model_key = data['model_name'] + '_' + '_'.join(str(x) for x in list(img.shape))
    img_tensor, _ = transfor(padded_img, None)
    img_tensor = img_tensor.unsqueeze(0).float().cuda()
    if model_key not in camera_models:
        model_path = [x for x in model_paths if x['name']==data['model_name']]
        if len(model_path)>0:
            model_path = model_path[0]['path']
        else:
            model_path = 'checkpoints/YoloV2/64/0,5-1,0-2,0/Coco_checkpoints.pth'
        model = torch.load(model_path).eval().cuda()
        trt_model = torch2trt(model, [img_tensor])
        trt_model.classes = model.classes
        trt_model.ranges =model.ranges
        camera_models[model_key] = trt_model
    outputs = camera_models[model_key](img_tensor)
    if len(outputs)==1:
        outputs = outputs,
    # if len(outputs.shape)==5:
    #     outputs = outputs.squeeze(0),
    outs = [camera_models[model_key].ranges.activate_output(out).squeeze(0).cpu().numpy() for out in outputs]

    pilImage = apply_output.apply_detections(target_to_box_transform, outs, [], Image.fromarray(img), camera_models[model_key].classes, 0.5)

    img = np.array(pilImage)[:img.shape[0], :img.shape[1]]
    img = cv2.resize(img, dsize=img_input.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    retval, buffer = cv2.imencode('.jpeg', img)
    data = {'image': 'data:image/png;base64,' + base64.b64encode(buffer).decode("utf-8") }

    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port="5001", threaded=False)