#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import least_squares
from models_4 import Line


def main():
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.grid()
    plt.xlabel('x')
    plt.xlabel('y')

    file = open('pts.pk1', 'rb')

    # dump information to that file
    data = pickle.load(file)
    file.close()
    
    plt.plot(data['xs'],data['ys'],'sk',linewidth=2,markersize=12)

    
    #Define the model
    line=Line(data)
    best_line=Line(data)

    result = least_squares(line.objectiveFunction,[line.m,line.b],verbose=2)



    

        

if __name__ == '__main__':
    main()

