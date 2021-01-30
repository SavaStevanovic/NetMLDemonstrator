import pybase64
import torch
from visualization import output_transform
from visualization import apply_output
from PIL import Image
from data_loader import augmentation
import tornado.web
import io
import torch.nn.functional as F
from prometheus_client import start_http_server, Summary

FILTERS_TIME = Summary('get_filters', 'Time spent processing filters')
MESSAGE_TIME = Summary('on_message', 'Time spent processing messages')

camera_models = {}
torch.set_grad_enabled(False)

model_paths = {
        "UNet" : {'path': 'checkpoints/Unet/64/ConvBlock/checkpoints_final.pth'},
        "DeepLabV3+" : {'path': 'checkpoints/DeepLabV3Plus/256/ResNetBackbone/256/3-4-6/checkpoints_final.pth'},
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

    @FILTERS_TIME.time()
    def get(self):
        data = {
            'models': list(self.model_paths.keys()),
        } 
      
        self.write(data)

class FrameUploadHandler(BaseHandler):
    @MESSAGE_TIME.time()
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
            model.preprocessing = augmentation.PairCompose([
                augmentation.PaddTransform(pad_size=2**model.depth),
                augmentation.OutputTransform()
            ])
            camera_models[model_key] = model.cuda()
        model = camera_models[model_key]
        img_tensor, _, _ = model.preprocessing(img_input, None, None)
        img_tensor = img_tensor.unsqueeze(0).float().cuda()
        output = model(img_tensor).detach()[0, 0, :img_input.size[1], :img_input.size[0]].sigmoid().cpu().numpy()

        mask_byte_arr = io.BytesIO()
        Image.fromarray((output*255).astype('uint8')).save(mask_byte_arr, format='jpeg')
        encoded_mask = 'data:image/jpeg;base64,' + pybase64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
        self.write(tornado.escape.json_encode(encoded_mask))

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    # 2. Create Tornado application
    app = tornado.web.Application([
        (r'/get_models', GetModelsHandler, dict(model_paths=model_paths)),
        (r'/frame_upload', FrameUploadHandler),] 
    )
    start_http_server(8000)
    server = tornado.httpserver.HTTPServer(app)
    server.listen(5005)
   
    tornado.ioloop.IOLoop.current().start()