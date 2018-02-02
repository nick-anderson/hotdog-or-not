import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import pickle
import os
import numpy as np
import cv2

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


os.system('curl %s -o image'%'http://del.h-cdn.co/assets/16/34/1600x800/landscape-1471884788-delish-zucchini-boats-lasagna.jpg')
img = cv2.imread('image',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(300,300), interpolation = cv2.INTER_CUBIC)
img = np.array(img).reshape(-1,300,300,1)
# pred = np.argmax(model.predict(img))
pred= model.predict(img)[0]
if pred[0] > .5:
	print('Nick says it is a Hot Dog with %s probability'%pred[0])
elif pred[1] >= .5:
	print('Nick says it is NOT Not Hot Dog with %s probability'%pred[1])
# if pred == 'hotdog':
	# print 
