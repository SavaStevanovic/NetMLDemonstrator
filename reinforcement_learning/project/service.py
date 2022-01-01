from os import listdir
import torch
from PIL import Image
from data_loader import augmentation
import tornado.web
import io
import torch.nn.functional as F
from prometheus_client import start_http_server, Summary
import os

FILTERS_TIME = Summary('get_filters', 'Time spent processing filters')
MESSAGE_TIME = Summary('on_message', 'Time spent processing messages')

camera_models = {}
torch.set_grad_enabled(False)

chackpoint_dir = "/app/tmp/checkpoints"

model_paths = [
    {
        "name": env,
        "models": os.listdir(os.path.join(chackpoint_dir, env))
    } for env in os.listdir(chackpoint_dir)
]


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.set_header("Access-Control-Allow-Headers",
                        "access-control-allow-origin,authorization,content-type")

    def options(self):
        # no body
        self.set_status(204)
        self.finish()


class GetModelsHandler(BaseHandler):
    def initialize(self, model_paths):
        self.model_paths = model_paths

    @tornado.gen.coroutine
    @FILTERS_TIME.time()
    def get(self):
        self.write(tornado.escape.json_encode(self.model_paths))


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
            model.target_output_transform = output_transform.PartAffinityFieldTransform(
                skeleton=model.skeleton, parts=model.parts, heatmap_distance=2)
            model.preprocessing = augmentation.PairCompose([
                augmentation.PaddTransform(pad_size=2**model.depth),
                augmentation.OutputTransform()
            ])
            camera_models[model_key] = model.cuda()
        model = camera_models[model_key]
        img_tensor, _, _, _ = model.preprocessing(img_input, None, None)
        img_tensor = img_tensor.unsqueeze(0).float().cuda()
        pafs_output, maps_output = model(img_tensor)
        pafs_output = F.interpolate(
            pafs_output[-1], img_input.size, mode='bicubic', align_corners=True)[0].detach().cpu().numpy()
        maps_output = F.interpolate(
            maps_output[-1], img_input.size, mode='bicubic', align_corners=True)[0].detach().cpu().numpy()
        outputs = model.target_output_transform(
            pafs_output, maps_output, data['bodypart'], data['joint']*2-1)
        return_msg = {}
        return_msg['parts'] = [
            (x[1][1]/img_input.size[1], x[1][0]/img_input.size[0]) for x in outputs[0]]
        return_msg['joints'] = [((x[0][1][1]/img_input.size[1], x[0][1][0]/img_input.size[0]),
                                 (x[1][1][1]/img_input.size[1], x[1][1][0]/img_input.size[0])) for x in outputs[1]]
        self.write(tornado.escape.json_encode(return_msg))


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    # 2. Create Tornado application
    app = tornado.web.Application([
        (r'/player/get_filters', GetModelsHandler, dict(model_paths=model_paths)),
        (r'/player/frame_upload', FrameUploadHandler), ]
    )
    start_http_server(8000)
    server = tornado.httpserver.HTTPServer(app)
    server.listen(4322)

    tornado.ioloop.IOLoop.current().start()
