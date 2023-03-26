import tornado
import time
import sockjs.tornado
import json
import asyncio
from prometheus_client import start_http_server, Summary

FILTERS_TIME = Summary('get_filters', 'Time spent processing filters')
MESSAGE_TIME = Summary('on_message', 'Time spent processing messages')

filter_data = {
    "detection": {'path': 'http://detection:5001/get_models'},
    "keypoint": {'path': 'http://keypoint:5004/get_models'},
    "segmentation": {'path': 'http://segmentation:5005/get_models'},
    "style": {'path': 'http://style:5009/get_models'}
}


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        print("setting headers!!!")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.set_header("Access-Control-Allow-Headers",
                        "access-control-allow-origin,authorization,content-type")

    def options(self):
        # no body
        self.set_status(204)
        self.finish()


class GetFilterHandler(BaseHandler):
    def initialize(self, filter_data):
        self.filter_data = filter_data.copy()

    @tornado.gen.coroutine
    @FILTERS_TIME.time()
    def get(self):
        remove_elems = []
        filters = [{
            'name': "Test",
            'models': ['Test_good', 'Test_bad'],
            'progress_bars':[
                {'name': 'bar_0.7', 'value': 0.7},
                {'name': 'bar_0.2', 'value': 0.2},
                {'name': 'bar_0.3', 'value': 0.3}],
            'check_boxes': [
                {'name': 'true_check', 'checked': True},
                {'name': 'false_check', 'checked': False}],
        }]

        for k, d in self.filter_data.items():
            http_client = tornado.httpclient.AsyncHTTPClient()
            http_client.defaults["connect_timeout"] = 0.05
            try:
                response = yield http_client.fetch(d['path'])
                if response.code == 200:
                    models = tornado.escape.json_decode(response.body)
                    models['name'] = k
                    filters.append(models)
            except Exception as e:
                remove_elems.append(k)
            http_client.defaults["connect_timeout"] = 20

        for k in remove_elems:
            self.filter_data.pop(k)
        self.write(tornado.escape.json_encode(filters))


class FrameUploadConnection(sockjs.tornado.SockJSConnection):
    def __init__(self, session):
        self.session = session
        self.filter_data = filter_data

    def on_open(self, info):
        pass

    @tornado.gen.coroutine
    @MESSAGE_TIME.time()
    def on_message(self, message):
        start_time = time.time()
        data = tornado.escape.json_decode(message)
        used_models = [x for x in data['config'] if 'selectedModel' in x
                                                    and x['selectedModel']
                                                    and x['name'] in filter_data.keys()]
        return_data = {}
        config_parameters = ['progress_bars', 'check_boxes']
        model_outputs = []
        http_client = tornado.httpclient.AsyncHTTPClient()
        for model_config in used_models:
            model_service = self.filter_data[model_config['name']]
            msg = {
                'frame': data['frame'],
                'model_name': model_config['selectedModel'],
            }
            for x in config_parameters:
                if x not in model_config:
                    msg[x] = []
                else:
                    msg[x] = model_config[x]
            r = tornado.httpclient.HTTPRequest(
                model_service['path'].replace('get_models', 'frame_upload'),
                body=json.dumps(msg),
                method="POST",
                headers={'model_name': model_config['name']})
            r = http_client.fetch(r)
            r.model_name = model_config['name']
            model_outputs.append(r)

        for future in asyncio.as_completed(model_outputs):
            try:
                result = yield future
                content = tornado.escape.json_decode(result.body)
                if 'detection' in result.effective_url:
                    return_data['bboxes'] = content
                if 'keypoint' in result.effective_url:
                    return_data['parts'] = content['parts']
                    return_data['joints'] = content['joints']
                if 'segmentation' in result.effective_url:
                    return_data['mask'] = content
                if 'style' in result.effective_url:
                    return_data['image'] = content
            except Exception as _:
                pass
        self.send(tornado.escape.json_encode(return_data))
        print("--- {} ms ---".format((time.time() - start_time)*1000))

    def on_close(self):
        pass


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    FrameRouter = sockjs.tornado.SockJSRouter(
        FrameUploadConnection, r'/menager/frame_upload_stream')
    app = tornado.web.Application([
        (r'/menager/get_filters', GetFilterHandler, dict(filter_data=filter_data)),
    ] + FrameRouter.urls
    )
    start_http_server(8000)
    server = tornado.httpserver.HTTPServer(app)
    server.listen(4321)

    tornado.ioloop.IOLoop.current().start()
