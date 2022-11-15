#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from scipy.optimize import least_squares
from models_5 import Sinusoid


def main():
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')

    file = open('pts.pk1', 'rb')

    # dump information to that file
    data = pickle.load(file)
    file.close()
    print(data)
    
    plt.plot(data['xs'],data['ys'],'sk',linewidth=2,markersize=6)

    
    #Define the model
    sinusoid=Sinusoid(data)
    sinusoid.randomize_params()

    result = least_squares(sinusoid.objectiveFunction,[sinusoid.a,sinusoid.b,sinusoid.h,sinusoid.k],verbose=2)        

if __name__ == '__main__':
    main()

