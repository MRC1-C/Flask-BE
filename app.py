from flask import Flask
import tensorflow as tf
import numpy as np
from flask import jsonify
from flask import request
from flask_cors import CORS

app = Flask(__name__)
app.debug = True
CORS(app)

model = tf.keras.models.load_model("model/LSTM")
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", None)
    predict = model.predict(np.array([text]))
    print(text)
    print(predict)
    res = {
        "label": str(1+predict.argmax())
    }
    return jsonify(res)

@app.route("/multipredict", methods=["POST"])
def multi_predict():
    res = []
    f = request.files
    for i in f:
        data = f[i].read().decode('utf-8').split('\n')
        r = {
            'name': i,
            "data": []
        }
        for j in data:
            if j!="" :
                predict = model.predict(np.array([j]))
                r["data"].append({
                    "text": j,
                    "label": str(1+predict.argmax())
                })
        res.append(r)
    return jsonify(res)