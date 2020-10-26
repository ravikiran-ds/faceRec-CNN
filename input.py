# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:21:08 2020

@author: HP
"""
from keras.preprocessing.image import ImageDataGenerator as IDG
from  capture import capture_photos

def inpu():
    #capture photo
    capture_photos("C:\\Users\\HP\\Documents\\ML\\Projects\\my face detection\\data\\test user input\\test_user\\",photos=5)
    #test with user input
    test_datagen=IDG(rescale=1./255)
    test_batches=test_datagen.flow_from_directory("C:\\Users\\HP\\Documents\\ML\\Projects\\my face detection\\data\\test user input\\",
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')
    return test_batches
