from flask import Flask
from flask import Flask, flash, request, redirect, render_template, make_response
from flask import Response, json, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
from google.cloud import firestore
import numpy as np

app = Flask(__name__)
app.secret_key = "secret_password_key_123456"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    return "ok"

@app.route('/upload', methods=['POST'])
def upload_file():
    print('.............')
    print(request.form['user_id'])
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            image_id = uuid.uuid4().hex
            user_id = request.form['user_id']

            print("before " + str(request.form))    
            admin = request.form['admin']
            image_path = "/home/sruthi/uploads/" + user_id + "/"
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            image_key = image_id + "." + file_extension
            image_full_path = os.path.join(image_path, image_key)
            file.save(image_full_path)

            image = cv2.imread(image_full_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model='hog')

            print(image_full_path)
            encodings = face_recognition.face_encodings(rgb, boxes)

            db = firestore.Client()
            image_encodings_ref = db.collection(u'image_encodings')
            encodings_dict = {str(idx) : i.tolist() for idx, i in enumerate(encodings)}
            image_encodings_ref.add({'image_id' : image_key, 'encodings' : encodings_dict, 'admin': admin, 'user_id' : user_id})
            js = json.dumps({'image_id' : image_key})
            resp = Response(js, status=200, mimetype='application/json')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

@app.route('/image/<upath>/<ipath>', methods=['GET'])
def show_image(upath, ipath):
    return send_file("/home/sruthi/uploads/" + upath + "/" + ipath)

@app.route('/compare_image/<ipath>', methods=['GET'])
def compare_image(ipath):
    db = firestore.Client()
    image_encodings_ref = db.collection(u'image_encodings')
    docs = image_encodings_ref.where("admin", "==", "false").where("image_id", "==", ipath).stream()
    current_image_encodings = list(next(docs).get('encodings').values())

    docs = image_encodings_ref.where("admin", "==", "true").stream()
    known_encodings = []
    known_image_ids = []

    for doc in docs:
        encodings = list(doc.get('encodings').values())
        image_id = doc.get('user_id') + '/' + doc.get('image_id')

        for encoding in encodings:
            known_encodings.append(encoding)
            known_image_ids.append(image_id)
    
    image_ids = []
    
    for encoding in current_image_encodings:
        matches = face_recognition.compare_faces(np.array(known_encodings), np.array(encoding))
        image_id = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                image_id = known_image_ids[i]
                counts[image_id] = counts.get(image_id, 0) + 1

            image_id = max(counts, key=counts.get)

        image_ids.append(image_id)
    
    js = json.dumps({'matches' : image_ids})
    resp = Response(js, status=200, mimetype='application/json')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    
