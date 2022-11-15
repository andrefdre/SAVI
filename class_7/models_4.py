#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

from random import uniform

class Line():
    def __init__(self,gt):
        self.gt = gt 
        self.randomize_params()
        self.first_draw =True

    def randomize_params(self):
        self.m=uniform(-2,2)
        self.b=uniform(-5,5)
    
    def getY(self,x):
        return self.m*x+self.b
    
    def objectiveFunction(self,params):
        self.m=params[0]
        self.b=params[1]
        residuals = []
        for gt_x,gt_y in zip(self.gt['xs'],self.gt['ys']):
            y=self.getY(gt_x)
            residual= abs(y-gt_y)
            residuals.append(residual)
        error =sum(residuals)
        print("Error" + str(error))
        self.draw()
        plt.waitforbuttonpress(0.1)
        return residuals
    
    def draw(self,color='b--'):
        xi=-10
        xf=10
        yy=self.getY(xi)
        yf=self.getY(xf)

        if self.first_draw:
            self.draw_handle = plt.plot([xi,xf],[yy,yf],color,linewidth=2)
            self.first_draw=False
        else:
            plt.setp(self.draw_handle,data=([xi,xf],[yy,yf]))
