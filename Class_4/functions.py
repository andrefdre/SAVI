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
        self.area = w * h

        self.x2=self.x1+self.w
        self.y2=self.y1+self.h

    def computeIOU(self, bbox2):
    
        x1_intr = min(self.x1, bbox2.x1)             
        y1_intr = min(self.y1, bbox2.y1)             
        x2_intr = max(self.x2, bbox2.x2)
        y2_intr = max(self.y2, bbox2.y2)

        w_intr = x2_intr - x1_intr
        h_intr = y2_intr - y1_intr
        A_intr = w_intr * h_intr
        #A_intr = max(0, x2_intr - x1_intr) * max(0, y2_intr - y1_intr )
        
        A_union = self.area + bbox2.area - A_intr
        
        return A_intr / float(A_union)

class Detection(Bounding_box):
    def __init__(self, x, y, w, h,img_gui,id):
        super().__init__(x, y, w, h)
        self.img_gui=img_gui
        self.id=id
        self.extractSmallImage(img_gui)

    def extractSmallImage(self, image_full):
        self.image = image_full[self.y1:self.y1+self.h, self.x1:self.x1+self.w]

    def draw_rectangle(self,color):
        cv2.rectangle(self.img_gui, (self.x1,self.y1), (self.x2,self.y2), (int(color[0]),int(color[1]),int(color[2])), 2, 1)
        cv2.putText(self.img_gui, 'DET ' + str(self.id), (self.x1,self.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[0]),int(color[1]),int(color[2])), 1, cv2.LINE_AA)


class Tracking():
    def __init__(self,detection,id):
        self.detections=[detection]
        self.id=id

    def draw_rectangle(self,img_gui,color):
        last_detection = self.detections[-1] # get the last detection
        cv2.rectangle(img_gui, (last_detection.x1,last_detection.y1), (last_detection.x2,last_detection.y2), (int(color[0]),int(color[1]),int(color[2])), 2, 1)
        cv2.putText(img_gui, 'TRK ' + str(self.id), (last_detection.x1+50,last_detection.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[0]),int(color[1]),int(color[2])), 1, cv2.LINE_AA)
   
    def addDetection(self, detection):
        self.detections.append(detection)

    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return 