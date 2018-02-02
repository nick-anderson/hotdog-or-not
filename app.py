from flask import Flask, abort, jsonify, request
from flasgger import Swagger,swag_from
import json
import requests
import numpy as np
import pickle
import urllib2
app = Flask(__name__)
swagger = Swagger(app)
@app.route('/predict/<chats>/')

def predict(chats,methods=['POST']):
    """Example endpoint final model returning a predictionn
    This is using docstringes for specifications.
    ---
    parameters:
      - name: chats
        in: path
        type: string
        required: true
        description: Do not use quotes, raw text accepted.
    definitions:
      Features:
        type: object
        properties:
          palette_name:
            type: array
            items:
              $ref: '#/definitions/nlp'
      Prediction:
        type: string
    responses:
      200:
        description: Class label
        schema:
          $ref: '#/definitions/features'
        examples:
          "AI is good :)": "Pos"
          "Very sad and upset about API": "neg"
    """
    os.system('curl %s -o image'%(urllib2.unquote(chats).decode('utf8')))
    return("5")


if __name__=='__main__':
  try :
    global model
  except:
    pass
  cnn = input_data(shape=[None, 300, 300, 1], name='input')
  cnn = conv_2d(cnn, 32, 5, activation='relu')
  cnn = max_pool_2d(cnn, 5)
  cnn = conv_2d(cnn, 64, 5, activation='relu')
  cnn = max_pool_2d(cnn, 5)
  cnn = conv_2d(cnn, 128, 5, activation='relu')
  cnn = max_pool_2d(cnn, 5)
  cnn = conv_2d(cnn, 64, 5, activation='relu')
  cnn = max_pool_2d(cnn, 5)
  cnn = conv_2d(cnn, 32, 5, activation='relu')
  cnn = max_pool_2d(cnn, 5)
  cnn = fully_connected(cnn, 1024, activation='relu')
  cnn = dropout(cnn, 0.8)
  cnn = fully_connected(cnn, 2, activation='softmax')
  cnn = regression(cnn, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')
  model = tflearn.DNN(cnn)
  model.load("model.tflearn")
  try:
    global model
  except:
    pass
	app.run(host='0.0.0.0',debug=True)
