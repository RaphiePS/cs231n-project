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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        # imaged = Image.open(BytesIO(decoded))
        # ar = np.asarray(imaged)

        # height = data["height"]
        # width = data["width"]
        # speed = data["speed"]
        # offset = data["offset"]
        # position = data["position"]
        # trafficOffsets = data["trafficOffsets"]
        # trafficPositions = data["trafficPositions"]
        # print num, speed, offset, position, trafficPositions[0], trafficOffsets[0]
        # minDist = min([abs(p - position) for p in trafficPositions])
        print ar.shape, data["reward"], data["terminal"]
        plt.imshow(ar, cmap = cm.Greys_r)
        plt.show()
        # img = Image.fromarray(ar, 'RGBA')
        # img.save('test%s.png' % num)

        # informally, in lane if 0.6 <= offset <= 0.8 or offset <= 0.1
        # we might wanna double-check this

        # one second forward pass
        # time.sleep(1)

        self.write(json.dumps({
            "keyLeft": False,
            "keyRight": False,
            "keyFaster": True, #speed < 5000,
            "keySlower": False, #speed > 5000,
        }))


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