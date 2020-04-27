# -*- coding: utf-8 -*-

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
#import tensorflow as tf

print(os.getcwd())

labels = ['Middle','Old','Young']

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'C:/Users/rohan/Desktop/Work/Age_detection_dataset/App/model/model.json'
MODEL_PATH2 = 'C:/Users/rohan/Desktop/Work/Age_detection_dataset/App/model/model.h5' 

model = load_model(MODEL_PATH2)
#model.load_weights(MODEL_PATH2)

model._make_predict_function()  

def model_predict(img_path,model):
    img = image.load_img(img_path, target_size=(64,64))
    
    img = np.array(img, dtype="float") / 255.0
    
    pred=model.predict(img)
    
    return pred



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        i=preds.argmax(axis=1)
        vals=np.amax(preds,axis=1)
        perc_vals = vals*100
        perc_vals_rounded = perc_vals.round(2)
        
        label_img = labels[i]
        
        result = label_img+": "+str(perc_vals_rounded)
        
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
