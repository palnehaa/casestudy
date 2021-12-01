# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:19:44 2021

@author: neha89
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('home.html', prediction_text='The predicted price of the house is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)