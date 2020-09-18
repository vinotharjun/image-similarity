from flask import Flask,jsonify,request
# from interface import *
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from model import prediction,transforms
import cv2
from numpy import dot
from numpy.linalg import norm
from flask_cors import CORS
from search import main_search
from flask import render_template

app = Flask(__name__)
CORS(app)




@app.route('/')
def hello():
    return 'Hello World!'


@app.route("/predict",methods=["POST"]) 
def predict():
    if "image1" not in request.files:
        return jsonify({
            "error":"image file is required"
        })
    
    #print(request.files.get("image"))
    try:
        filestr = request.files['image1'].read()
        filestr2 = request.files['image2'].read()
        typ = request.form["type"]
        if typ is None:
            typ = "tripplet"
        npimg = np.fromstring(filestr, np.uint8)
        npimg2 = np.fromstring(filestr2, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(img)
        img_pil2 = Image.fromarray(img2)
        output1= prediction(img_pil,model_type=typ).squeeze().cpu().detach().numpy()
        output2= prediction(img_pil2,model_type=typ).squeeze().cpu().detach().numpy()
        # print(cosine_similarity(output1,output2))
        value = cosine_similarity(output1,output2)
        if value>0.5:
            result = "same image"
        else:
            result = "different image"
        return jsonify({
            "similarity_score":str("Result :"+result+" "+" , \n\n Cosine Similarity Value :"+str(value))
        })
    except Exception as e:
        print("error in deploy.py line 63",str(e))
        return jsonify({
            "error":"something went wrong with the uploaded file"
        })
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def cosine_similarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim
@app.route("/search",methods=["POST"])
def search():
    if "image" not in request.files:
        return jsonify({
            "error":"image file is required"
        })
    try:
        image_file = request.files["image"].read()
        npimg = np.fromstring(image_file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(img)
        image_feature = prediction(img_pil).squeeze().cpu().detach().numpy()
        close_list = main_search(image_feature)
        print(len(close_list))
        ret =[]
        for i in close_list:
            ret.append(i["image"])
        # print(ret)
        return jsonify({
            "list": ret
        })


        # return render_template('template.html', image_name=ret)
    except Exception as e:
        print("error in deploy.py line 41",str(e))
        return jsonify({
            "error":"something went wrong with the uploaded file"
        })

if __name__ == '__main__':
    app.run(threaded=False,port=5000)