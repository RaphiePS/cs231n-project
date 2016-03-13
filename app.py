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
        # self.redirect("/static/v2.curves.html")
        self.redirect("/static/v4.final.html")

class FrameHandler(tornado.web.RequestHandler):
    def post(self):
        data = json.loads(self.get_arguments("telemetry")[0])
        ar = np.fromstring(base64.decodestring(self.request.body), dtype=np.uint8)
        image = ar.reshape(hp.INPUT_SIZE, hp.INPUT_SIZE, hp.NUM_CHANNELS)
        left, right, faster, slower = data["action"]
        terminal, action, all_data, was_start = (
            data["terminal"],
            Action(left=left, right=right, faster=faster, slower=slower),
            data["all_data"],
            data["was_start"]
        )

        print "FRAME NUM: %d" % (agent.frame_count + 1), terminal, action.to_dict(), was_start

        # all_data = {collision, position, speed, max_speed} x 4 
        # abs(position) > 0.8 implies off road
        reward = agent.reward(all_data)
        print "Current reward = %.2f" % reward

        result_action = agent.step(image=image, reward=reward, terminal=terminal, was_start=was_start, action=action, telemetry=all_data)
        self.write(json.dumps(result_action.to_dict()))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/frame", FrameHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path})
    ], debug=True)

if __name__ == "__main__":
    app = make_app()
    if "SERVER_PORT" in os.environ:
        port = int(os.environ["SERVER_PORT"])
    else:
        port = 8000
    print "LISTENING ON PORT: %d" % port
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()
