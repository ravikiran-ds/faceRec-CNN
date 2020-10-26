# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:02:45 2020

@author: HP
"""
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np

#initalize cnn
cnn_cls=Sequential()
#adding convolution layer
cnn_cls.add(Convolution2D(input_shape=(64,64,3),kernel_size=(3,3),filters=32))
#adding maxpooliing layer
cnn_cls.add(MaxPooling2D(pool_size=(2,2)))
#flattening
cnn_cls.add(Flatten())
#full connection
cnn_cls.add(Dense(output_dim=128,input_dim=1,activation='relu',init='uniform'))
cnn_cls.add(Dense(output_dim=1,activation='sigmoid'))
#compiling cnn
cnn_cls.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#image preprocessing
from keras.preprocessing.image import ImageDataGenerator as IDG
train_datagen=IDG(rescale=1./255,
                  shear_range=0.2,
                  zoom_range=0.2,
                  horizontal_flip=True)

test_datagen=IDG(rescale=1./255)

train_generator=train_datagen.flow_from_directory("train\\path",
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

test_generator=test_datagen.flow_from_directory("test\\path",
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

#fitting the data
cnn_cls.fit_generator(train_generator,steps_per_epoch=len(train_generator),nb_epoch=3,validation_data=test_generator,
                      nb_val_samples=len(test_generator))

#saving the model
cnn_cls.save('face detection.model')





