import os
import time
from flask import Flask, abort, request, jsonify, g, url_for, render_template
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from flask_sqlalchemy import SQLAlchemy
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import config
import json
from engine.basic_img_classification import train, predict

app = Flask(__name__)

app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URL
app.config['upload'] = config.UPLOAD
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db = SQLAlchemy(app)
auth = HTTPBasicAuth()


## RECORD MODEL
class Record(db.Model):
    __tablename__ = 'records'
    id = db.Column(db.Integer, primary_key=True)
    prediction_time = db.Column(db.DateTime)
    image_link = db.Column(db.String(512))
    image_label = db.Column(db.String(128))


@app.route('/')
def index():
    return render_template("index.html", greeting = "Hello, ArtLabs")


## POST /api/train
@app.route('/api/train', methods=['POST'])
def train_engine():
    try:
        train()
        status = True
    except:
        status = False
        return jsonify({"success": status, "msg": "Failed to train."}), 404
    return jsonify({"success": status}), 200


## GET /api/predict
@app.route('/api/predict', methods=['GET'])
def predict_engine():
    try:
        image_link = request.args['image_link']
        if image_link == "":
          return jsonify({"success": False, "msg": "img_link is empty."}), 404
    except:
        return jsonify({"success": False, "msg": "img_link is required."}), 404

    # predict function
    image_label = predict(image_link)
    print(image_label)
    record = Record(image_link = image_link, image_label = image_label, prediction_time = datetime.now())
    db.session.add(record)
    db.session.commit()
    return jsonify({"prediction": image_label}), 200


## GET /api/get_past_predictions/
@app.route('/api/get_past_predictions', methods=['GET'])
def get_past_predictions():
    record = Record.query.all()
    result = [ {"prediction_time": p.prediction_time, "image_link": p.image_link, "image_label": p.image_label } for p in record]
    return jsonify({"predictions": result}), 200


## DELETE /api/clear_past_predictions
@app.route('/api/clear_past_predictions', methods=['DELETE'])
def clear_past_predictions():
    Record.query.delete()
    db.session.commit()
    return "" , 204


if __name__ == '__main__':
    if not os.path.exists(config.DATABASE_NAME):
        db.create_all()
    # configuration()
    app.run(debug=True, port='3035')