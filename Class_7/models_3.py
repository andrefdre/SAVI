#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import math

class Sinusoid():
    def __init__(self,gt):
        self.gt = gt 
        self.randomize_params()
        self.first_draw =True
        self.xs_for_plot= list(np.linspace(-10,10,num=500))

    def randomize_params(self):
        self.a=uniform(-10,10)
        self.b=uniform(-10,10)
        self.h=uniform(-10,10)
        self.k=uniform(-10,10)
    
    def getY(self,x):
        return self.a*math.sin(self.b*(x-self.h)+self.k)

    def getYs(self,xs):
        #REtrieves a list of ys
        ys=[]
        for x in xs:
            ys.append(self.getY(x))
        return ys

    def objectiveFunction(self):
        residuals = []
        for gt_x,gt_y in zip(self.gt['xs'],self.gt['ys']):
            y=self.getY(gt_x)
            residual= abs(y-gt_y)
            residuals.append(residual)

        error =sum(residuals)
        return error
    
    def draw(self,color='b--'):
        xi=-10
        xf=10
        yy=self.getY(xi)


        if self.first_draw:
            self.draw_handle = plt.plot(self.xs_for_plot,self.getYs(self.xs_for_plot),color,linewidth=2)
            self.first_draw=False
        else:
            plt.setp(self.draw_handle,data=(self.xs_for_plot,self.getYs(self.xs_for_plot)))
