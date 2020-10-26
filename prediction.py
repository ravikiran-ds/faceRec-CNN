# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:42:55 2020

@author: HP
"""
from input import inpu
from keras.models import load_model
from speak import SpeakText
speak=SpeakText()

def pred():
    rav_pred=[[]]
    #getting input and storing ing test
    test=inpu()
    #loading the model
    model=load_model("face detection.model")
    #predicting with model
    pred=model.predict(x=test,steps=len(test),verbose=0)
    rav_pred=pred
    return rav_pred

def out():
    p=pred()
    #print(p.mean())
    if p.mean()>0.8:
        speak.Speak("Hello Ravi!",speed=150,sound=1)
    else:
        speak.Speak("Hello Ma",speed=150,sound=1)
    
out()