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

    def extractSmallImage(self, image_full):
        return  image_full[self.y1:self.y1+self.h, self.x1:self.x1+self.w]


class Detection(Bounding_box):
    def __init__(self, x, y, w, h,img_full,id):
        super().__init__(x, y, w, h)
        self.id=id
        self.image = self.extractSmallImage(img_full)
        self.tracker_assigned=False


    def draw(self, image_gui, color=(255,0,0)):
        cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)

        image = cv2.putText(image_gui, 'D' + str(self.id), (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

class Tracker():
    def __init__(self,detection,id):
        self.detections=[detection]
        self.id=id
        self.Template=None
        self.bboxes = []

    def draw_last_detection(self,img_gui,color=(255,0,255)):
        last_detection = self.detections[-1] # get the last detection
        cv2.rectangle(img_gui, (last_detection.x1,last_detection.y1), (last_detection.x2,last_detection.y2), (int(color[0]),int(color[1]),int(color[2])), 2, 1)
        cv2.putText(img_gui, 'TRK ' + str(self.id), (last_detection.x1+50,last_detection.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
   
    def draw(self,img_gui,color=(255,0,255)):
        bbox= self.bboxes[-1]
        cv2.rectangle(img_gui, (bbox.x1,bbox.y1), (bbox.x1+bbox.w,bbox.y2+bbox.h), (int(color[0]),int(color[1]),int(color[2])), 2, 1)
        cv2.putText(img_gui, 'TRK ' + str(self.id), (bbox.x1+50,bbox.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
   
    
    def addDetection(self, detection):
        self.detections.append(detection)
        detection.tracker_assigned=True
        self.Template=detection.image
        bbox= Bounding_box(detection.x1,detection.y1,detection.h,detection.w)
        self.bboxes.append(bbox)

    def track(self,image):
        h,w=self.Template.shape
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(image_gray,self.Template,cv2.TM_CCOEFF_NORMED)
        _,max_val , _ , max_loc = cv2.minMaxLoc(result)

        x1=max_loc[0]
        y1=max_loc[1]

        self.Template=bbox.extractSmallImage(image)

        bbox= Bounding_box(x1,y1,h,w)
        self.bboxes.append(bbox)

    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text