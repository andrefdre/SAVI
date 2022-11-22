#!/usr/bin/env python3

import numpy as np
import cv2
from copy import deepcopy

class ImageMosaic():
    def __init__(self,q_image,t_image):
        self.q_image= q_image
        self.t_image=t_image
        self.q_image=self.q_image.astype(float)/255.0
        self.t_image=self.t_image.astype(float)/255.0
        self.overlap_mask= q_image[:,:,0]>0
        self.q_height,self.q_width , _ = q_image.shape
        self.t_height,self.t_width , _ = t_image.shape
        self.randomize_params()


    def randomize_params(self):
        # self.q_scale=uniform(-10,10)
        # self.q_bias=uniform(-10,10)
        # self.t_scale=uniform(-10,10)
        # self.q_scale=uniform(-10,10)
        self.q_scale=1.0
        self.q_bias=0.0
        self.t_scale=1.0
        self.t_bias=0.0
    
    def correct_images(self):
        # Correct images with the parameters
        self.q_image_c=self.q_image*self.q_scale+self.q_bias
        self.q_image_c[self.q_image_c>1]=1 # oversaturate at 1
        self.q_image_c[self.q_image_c<0]=0 #under saturate at 0
        self.t_image_c=self.t_image*self.t_scale+self.t_bias
        self.t_image_c[self.t_image_c>1]=1 # oversaturate at 1
        self.t_image_c[self.t_image_c<0]=0 #under saturate at 0


    def objectiveFunction(self,params):
        self.q_scale=params[0]
        self.q_bias=params[1]
        self.t_scale=params[2]
        self.t_bias=params[3]
        self.correct_images()
        residuals = []
        #Matrix form alternative
        diffs=np.abs(self.t_image_c-self.q_image_c)
        diffs_in_overlap=diffs[self.overlap_mask]
        residuals=np.sum(diffs_in_overlap)
        error =np.sum(residuals)
        print("Error" + str(error))
        self.draw()
        return residuals

    def drawFloatImage(self,win_name,image_f):
        image_uint8=(image_f*255).astype(np.uint8)   
        cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name,600,400)
        cv2.imshow(win_name, image_uint8) 

    def draw(self):
        stiched_image=deepcopy(self.t_image_c)
        stiched_image[self.overlap_mask]=(self.q_image_c[self.overlap_mask]+self.t_image_c[self.overlap_mask])/2
        self.drawFloatImage('asd',stiched_image)
        cv2.waitKey(20)