# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:15:14 2021

@author: lok bahadur chhetri
"""
from flask import Flask,request
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in =open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return'This is a project for Bank note detection'

@app.route('/predict')
def predict_bank_note():
    """
    Let's authenticate the bank note
    This is using docstring for specification

    -------
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
        
       

    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "the prediction is"+str(prediction)

@app.route('/predict_file',methods = ["POST"])
def predict_bank_file():
    """Let's Authenticate the bank notes
    This is using docstrings for specification
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
          
    responses:
        200:
            description: The output values
    """    
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "the prediction is"+str(list(prediction))


if __name__ == "__main__":
    app.run()