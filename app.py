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
from agent import Agent
from action import Action
from PIL import Image
from io import BytesIO

static_path = os.path.join(os.getcwd(), "static")
agent = Agent()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.redirect("/static/v4.final.html")


class FrameHandler(tornado.web.RequestHandler):
    def post(self):
        data = json.loads(self.request.body)
        ar = np.fromstring(base64.decodestring(data["image"]), dtype=np.uint8)
        image = ar.reshape(hp.INPUT_SIZE, hp.INPUT_SIZE)
        left, right, faster, slower = data["action"]
        terminal, action, reward, was_start = (
            data["terminal"],
            Action(left=left, right=right, faster=faster, slower=slower),
            data["reward"],
            data["was_start"]
        )

        print terminal, action.to_dict(), reward, was_start

        # TODO
        result_action = agent.step(image=image, reward=reward, terminal=terminal, was_start=was_start, action=action)
        self.write(json.dumps(result_action.to_dict()))


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