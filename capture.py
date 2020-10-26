# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:28:29 2020

@author: HP
"""
import cv2

def capture_photos(path,photos=1000):
    cap= cv2.VideoCapture(0)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if ret == False:
            break
        cv2.imwrite(path+str(i)+'.jpg',frame)
        cv2.imshow('Input', frame)
        i+=1
        if i==photos:
            break
        c = cv2.waitKey(1)
        if c == 27:   #press esc key to stop
            break
    cap.release()
    cv2.destroyAllWindows()

 #thats it






