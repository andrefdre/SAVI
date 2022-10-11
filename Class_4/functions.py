#!/usr/bin/env python3


import numpy as np
import cv2
import csv

class Bounding_box():
    def __init__(self,x,y,w,h):
        self.x1=x
        self.y1=y
        self.h=h
        self.w=w

        self.x2=self.x1+self.w
        self.y2=self.y1+self.h

class Detection(Bounding_box):
    def __init__(self, x, y, w, h,img_gui,id):
        super().__init__(x, y, w, h)
        self.img_gui=img_gui
        self.id=id

    def draw_rectangle(self,color):
        cv2.rectangle(self.img_gui, (self.x1,self.y1), (self.x2,self.y2), (int(color[0]),int(color[1]),int(color[2])), 2, 1)
        cv2.putText(self.img_gui, 'ID ' + str(self.id), (self.x1,self.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[0]),int(color[1]),int(color[2])), 1, cv2.LINE_AA)


class Tracking():
    def __init__(self,detection,id):
        self.detection=detection
        self.id=id

    def draw_rectangle(self,img_gui,color):
        cv2.rectangle(img_gui, (self.detection.x1,self.detection.y1), (self.detection.x2,self.detection.y2), (int(color[0]),int(color[1]),int(color[2])), 2, 1)
        cv2.putText(img_gui, 'ID ' + str(self.id), (self.detection.x1,self.detection.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[0]),int(color[1]),int(color[2])), 1, cv2.LINE_AA)
