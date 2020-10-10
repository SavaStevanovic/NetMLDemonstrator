import pybase64
import torch
from visualization import output_transform
from visualization import apply_output
from PIL import Image
from data_loader import augmentation
import tornado.web
import io


camera_models = {}

model_paths = {
        "YoloV2" : {'path': 'checkpoints/YoloV2/64/0,5-1,0-2,0/Coco_checkpoints_final.pth'},
        "Yolo" : {'path': 'checkpoints/YoloNet/512/0,5-1,0-2,0/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        "RetinaNet" : {'path': 'checkpoints/RetinaNet/512/0,5-1,0-2,0/FeaturePyramidBackbone/512/ResNetBackbone/512/3-4-6-3/Coco_checkpoints_final.pth'},
        "FPN" : {'path': 'checkpoints/FeaturePyramidNet/2048/0,5-1,0-2,0/FeaturePyramidBackbone/2048/Coco_checkpoints_final.pth'},
    }

class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
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
            'progress_bars':[{'name':'threshold',  'value':0.5}], 
            'check_boxes': [{'name':'NMS', 'checked':True}],
        } 
      
        self.write(data)

class FrameUploadHandler(BaseHandler):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        for d in data['progress_bars']:
            data[d['name']] = d['value']
        for d in data['check_boxes']:
            data[d['name']] = d['checked']
        image_data = data['frame'].replace('data:image/jpeg;base64,', "")
        byte_image = io.BytesIO(pybase64._pybase64.b64decode(image_data))
        img_input = Image.open(byte_image)

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
            model.preprocessing = augmentation.PairCompose([
                augmentation.PaddTransform(pad_size=2**model.depth),
                augmentation.OutputTransform()
            ])
            camera_models[model_key] = model
        model = camera_models[model_key]
        img_tensor, _  = model.preprocessing(img_input, None)
        img_tensor = img_tensor.unsqueeze(0).float().cuda()
        outputs = model(img_tensor)
        outs = [out.cpu().detach().numpy() for out in outputs]

        boxes_pr=[]
        for i, out in enumerate(outs):
            boxes_pr += model.target_to_box_transform(out, data['threshold'], scale=img_input.size[::-1], depth = i)
        if data['NMS']:
            boxes_pr = apply_output.non_max_suppression(boxes_pr)
         
        for b in boxes_pr:
            b['class'] = model.classes[b['category_id']][1]
        self.write(tornado.escape.json_encode(boxes_pr))

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    # 2. Create Tornado application
    app = tornado.web.Application([
        (r'/get_models', GetModelsHandler, dict(model_paths=model_paths)),
        (r'/frame_upload', FrameUploadHandler),] 
    )

    server = tornado.httpserver.HTTPServer(app)
    server.listen(5001)
   
    tornado.ioloop.IOLoop.current().start()