"""
    Simple sockjs-tornado chat application. By default will listen on port 8080.
"""
import tornado.ioloop
import tornado.web
import json 
import base64
import cv2
import numpy as np
import requests

import sockjs.tornado
filter_data = {
    "detection": {'path': 'http://detection:5001/get_models'}
}

class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        print("setting headers!!!")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.set_header("Access-Control-Allow-Headers", "access-control-allow-origin,authorization,content-type") 

    def options(self):
        # no body
        self.set_status(204)
        self.finish()

class GetFilterHandler(BaseHandler):
    def initialize(self, filter_data):
        self.filter_data = filter_data
    
    async def get(self):
        filters = [{
            'name': "Test", 
            'models': ['Test_good', 'Test_bad'], 
            'progress_bars':[
                {'name':'bar_0.7', 'value':0.7}, 
                {'name':'bar_0.2', 'value':0.2}, 
                {'name':'bar_0.3', 'value':0.3}], 
            'check_boxes': [
                {'name':'true_check', 'checked':True}, 
                {'name':'false_check', 'checked':False}],
            }]

        for k, d in self.filter_data.items():
            http_client = tornado.httpclient.AsyncHTTPClient()
            try:
                response = await http_client.fetch(d['path'])
                if response.code == 200:
                    models = json.loads(response.body.decode("utf-8")) 
                    models['name'] = k
                    filters.append(models)
            except Exception as e:
                pass
        self.write(json.dumps(filters))

class FrameUploadHandler(BaseHandler):
    def initialize(self, filter_data):
        self.filter_data = filter_data

    async def post(self):
        data = json.loads(self.request.body.decode("utf-8"))
        used_models = [x for x in data['config'] if 'selectedModel' in x and x['name'] in filter_data.keys()]
        for model_config in used_models:
            model_service = self.filter_data[model_config['name']]
            headers = {'Content-Type': 'application/json'}

            http_client = tornado.httpclient.AsyncHTTPClient()
            data['model_name'] = model_config['selectedModel']
            response = await http_client.fetch(
                request=model_service['path'].replace('get_models', 'frame_upload'),
                method='POST',
                body=json.dumps(data),
                headers=headers,
                )
            self.write(response.body)
            return

        image_data = data['frame'].replace('data:image/png;base64,', "")
        byte_image = bytearray(base64.b64decode(image_data))
        frame = cv2.imdecode(np.asarray(byte_image), cv2.IMREAD_COLOR)
        ret_data = {'image': data['frame']}

        self.write(json.dumps(ret_data))

class FrameUploadConnection(sockjs.tornado.SockJSConnection):
    def __init__(self, session):
        self.session = session
        self.filter_data = filter_data

    def on_open(self, info):
        pass

    def on_message(self, message):
        data = json.loads(message)
        used_models = [x for x in data['config'] if 'selectedModel' in x and x['name'] in filter_data.keys()]
        for model_config in used_models:
            model_service = self.filter_data[model_config['name']]
            headers = {'Content-Type': 'application/json'}
            data['model_name'] = model_config['selectedModel']
            r = requests.post(url=model_service['path'].replace('get_models', 'frame_upload'),json = {
                    'frame': data['frame'], 
                    'model_name': model_config['selectedModel'],
                    'progress_bars' : model_config['progress_bars'],
                    'check_boxes' : model_config['check_boxes']
                })
            self.send(r.content)
            return

        image_data = data['frame'].replace('data:image/png;base64,', "")
        byte_image = bytearray(base64.b64decode(image_data))
        frame = cv2.imdecode(np.asarray(byte_image), cv2.IMREAD_COLOR)
        ret_data = {'image': data['frame']}

        self.send(json.dumps(ret_data))

    def on_close(self):
        pass

class MessageConnection(sockjs.tornado.SockJSConnection):
    """Chat connection implementation"""
    # Class level variable
    def __init__(self, session):
        """Connection constructor.

        `session`
            Associated session
        """
        self.session = session
    def on_open(self, info):
        pass

    def on_message(self, message):
        self.send(message)

    def on_close(self):
        pass

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    # 1. Create chat router
    ChatRouter = sockjs.tornado.SockJSRouter(MessageConnection, '/echo', )

    FrameRouter = sockjs.tornado.SockJSRouter(FrameUploadConnection, '/frame_upload_stream')
    # 2. Create Tornado application
    app = tornado.web.Application([
        (r'/get_filters', GetFilterHandler, dict(filter_data=filter_data)),
        (r'/frame_upload', FrameUploadHandler, dict(filter_data=filter_data)),] 
        + ChatRouter.urls
        + FrameRouter.urls)

    # 3. Make Tornado app listen on port 8080
    app.listen(4321)
    
    # 4. Start IOLoop
    tornado.ioloop.IOLoop.current().start()