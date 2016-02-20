import tornado.ioloop
import tornado.web
from tornado import gen
import os
import base64
import re
import json
import numpy as np
import time
from PIL import Image

static_path = os.path.join(os.getcwd(), "static")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

class FrameHandler(tornado.web.RequestHandler):
    def post(self, num):
        # j = json.loads(self.request.body)
        ar = np.fromstring(self.request.files["image"][0]["body"], dtype="uint8")
        height = int(self.get_arguments("height")[0])
        width = int(self.get_arguments("width")[0])
        ar = ar.reshape(height, width, 4)
        speed = float(self.get_arguments("speed")[0])
        print num, speed, ar.shape
        # img = Image.fromarray(ar, 'RGBA')
        # img.save('test%s.png' % num)
        self.write(json.dumps({
            "keyLeft": False,
            "keyRight": False,
            "keyFaster": speed < 5000,
            "keySlower": speed > 5000,
        }))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/frame/(.*)", FrameHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path})
    ], debug=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()