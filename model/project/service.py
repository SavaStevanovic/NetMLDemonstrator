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
import sockjs.tornado
import tornado.ioloop
import tornado.web


transfor = augmentation.OutputTransform()
camera_models = {}

model_paths = {
        "YoloV2" : {'path': 'checkpoints/YoloV2/64/0,5-1,0-2,0/Coco_checkpoints_final.pth'},
        "Yolo" : {'path': 'checkpoints/YoloNet/512/0,5-1,0-2,0/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        "RetinaNet" : {'path': 'checkpoints/RetinaNet/512/0,5-1,0-2,0/FeaturePyramidBackbone/512/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        "FPN" : {'path': 'checkpoints/FeaturePyramidNet/2048/0,5-1,0-2,0/FeaturePyramidBackbone/2048/Coco_checkpoints_final.pth'},
    }

class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        # print("setting headers!!!")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.set_header("Access-Control-Allow-Headers", "access-control-allow-origin,authorization,content-type") 

    def options(self):
        # no body
        self.set_status(204)
        self.finish()

class GetModelsHandler(BaseHandler):
    def initialize(self, model_paths):
        self.model_paths = model_paths
    
    def get(self):
        data = {
            'models': list(self.model_paths.keys()),
            'progress_bars':[{'name':'threshold',  'value':0.7}], 
            'check_boxes': [{'name':'NMS', 'checked':True}],
        } 
      
        self.write(json.dumps(data))

class FrameUploadHandler(BaseHandler):
    def post(self):
        data = json.loads(self.request.body.decode("utf-8"))
        for d in data['progress_bars']:
            data[d['name']] = d['value']
        for d in data['check_boxes']:
            data[d['name']] = d['checked']
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
            ], Image.fromarray(img), model.classes, data['threshold'], data['NMS'])

        img = cv2.resize(img, dsize=img_input.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        retval, buffer = cv2.imencode('.jpeg', img)
        data = {'image': 'data:image/png;base64,' + base64.b64encode(buffer).decode("utf-8")}

        self.write(json.dumps(data))

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    # 2. Create Tornado application
    app = tornado.web.Application([
        (r'/get_models', GetModelsHandler, dict(model_paths=model_paths)),
        (r'/frame_upload', FrameUploadHandler),] 
    )

    # 3. Make Tornado app listen on port 8080
    app.listen(5001)
    
    # 4. Start IOLoop
    tornado.ioloop.IOLoop.current().start()