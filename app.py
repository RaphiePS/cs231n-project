import tornado.ioloop
import tornado.web
from tornado import gen
import os
import base64
import re
import json
import numpy as np
import time
import hyperparameters as hp
import agent
from PIL import Image
from io import BytesIO

static_path = os.path.join(os.getcwd(), "static")


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.redirect("/static/v4.final.html")


class FrameHandler(tornado.web.RequestHandler):
    def post(self):
        data = json.loads(self.request.body)
        img = data["image"]
        ar = np.fromstring(base64.decodestring(img), dtype=np.uint8)
        ar = ar.reshape(hp.INPUT_SIZE, hp.INPUT_SIZE)
        action = agent.step(image=ar, reward=data["reward"], terminal=data["terminal"])


        self.write(json.dumps(action.to_dict()))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/frame", FrameHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path})
    ], debug=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()