from os import listdir
import time
import torch
from PIL import Image
from data_loader import augmentation
import tornado.web
import io
import torch.nn.functional as F
from prometheus_client import start_http_server, Summary
import os
import sockjs.tornado
import environment.playgrounds as play
import algorithams
import pybase64
import numpy as np
FILTERS_TIME = Summary('get_filters', 'Time spent processing filters')
MESSAGE_TIME = Summary('on_message', 'Time spent processing messages')

models = {}
torch.set_grad_enabled(False)

chackpoint_dir = "/app/tmp/checkpoints"

env_map = {x.__name__: x for x in
           [
               play.AcrobotV1,
               play.BreakoutV0,
               play.CartPoleV0,
               play.CartPoleV1,
               play.LunarLanderContinuousV2,
               play.LunarLanderV2,
               play.MountainCarV0,
               play.PendulumV1,
           ]
           }

alg_map = {x.__name__: x for x in
           [
               algorithams.ppo.PPO,
               algorithams.actor_critic.A2C,
               algorithams.policy_gradient.PolicyGradient,
               algorithams.dqn.DQN,
           ]
           }

model_paths = [
    {
        "name": env,
        "models": os.listdir(os.path.join(chackpoint_dir, env))
    } for env in os.listdir(chackpoint_dir) if env in env_map
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


class FrameUploadConnection(sockjs.tornado.SockJSConnection):
    def __init__(self, session):
        self.session = session
        self.model_paths = model_paths
        self.environment = None
        self.alg = None

    @tornado.gen.coroutine
    @MESSAGE_TIME.time()
    def on_message(self, message):
        if self.environment is None:
            data = tornado.escape.json_decode(message)
            used_models = [x for x in data['config'] if 'selectedModel' in x
                                                        and x['selectedModel']
                                                        and x['name'] in [x["name"] for x in self.model_paths]]
            if not used_models:
                return
            used_model = used_models[0]
            environment = env_map[used_model["name"]](False)
            algoritham_class = alg_map[used_model["selectedModel"]]
            alg = algoritham_class(
                env=environment,
                inplanes=64,
                block_counts=[],
                input_size=environment.env.observation_space,
                output_size=environment.env.action_space
            )
            alg.load_best_state()
            state = environment.reset()
            self.environment = environment
            self.alg = alg
            # Select and perform an action
        action, _ = self.alg.preform_action(
            np.float32(self.environment.env.state))
        state, _, done, _ = self.environment.step(action)
        frame = self.environment.env.render(
            mode='rgb_array')
        mask_byte_arr = io.BytesIO()
        Image.fromarray(frame).save(
            mask_byte_arr, format='jpeg')
        encoded_mask = 'data:image/jpeg;base64,' + \
            pybase64.b64encode(
                mask_byte_arr.getvalue()).decode('utf-8')
        self.send(tornado.escape.json_encode(encoded_mask))
        if done:
            self.close()

    def on_close(self):
        if self.environment:
            self.environment.env.close()
        print("close")


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    FrameRouter = sockjs.tornado.SockJSRouter(
        FrameUploadConnection, r'/player/frame_upload_stream')
    app = tornado.web.Application([
        (r'/player/get_filters', GetModelsHandler, dict(model_paths=model_paths)),
    ] + FrameRouter.urls
    )
    start_http_server(8000)
    server = tornado.httpserver.HTTPServer(app)
    server.listen(4322)
    server.start(2)
    tornado.ioloop.IOLoop.current().start()
