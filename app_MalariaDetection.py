#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# In[2]:


# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[3]:


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer


# In[4]:


# Define a flask app
app = Flask(__name__)


# In[5]:


# Model saved with Keras model.save()
MODEL_PATH ='model_vgg19.h5'


# In[6]:


# Load your trained model
model = load_model(MODEL_PATH)


# In[ ]:





# In[ ]:





# In[7]:


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    
    
     # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With Pneumonia"
    else:
        preds="The Person is not Infected With Pneumonia"
    
    
    return preds


# In[8]:


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# In[9]:


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
        result=preds
        return result
    return None


# In[11]:


if __name__ == '__main__':
#     app.run(debug=True)
    app.run(debug=True)


# In[ ]:




