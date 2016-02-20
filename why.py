from flask import Flask
import json
app = Flask(__name__)

@app.route("/frame/<num>", methods=["POST"])
def frame(num):
    print num
    return json.dumps({
        "keyLeft": False,
        "keyRight": False,
        "keyFaster": True,
        "keySlower": False,
    })

if __name__ == "__main__":
    app.run()