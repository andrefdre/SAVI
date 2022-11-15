#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from models_3 import Sinusoid


def main():
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.grid(True)
    plt.xlabel('x')
    plt.xlabel('y')

    file = open('pts.pk1', 'rb')

    # dump information to that file
    data = pickle.load(file)
    file.close()

    print("Created a figure")
    
    plt.plot(data['xs'],data['ys'],'rx',linewidth=2,markersize=12)

    
    #Define the model
    sinusoid=Sinusoid(data)
    best_model=Sinusoid(data)
    best_error=1E6


    # Execution
    while True:
        #set new values
        sinusoid.randomize_params()
        # compute error
        error=sinusoid.objectiveFunction()

        if error < best_error:
            best_model.a=sinusoid.a
            best_model.b=sinusoid.b
            best_model.h=sinusoid.h
            best_model.k=sinusoid.k
            best_error=error
            print("We found a better model")

        print("Error: " ,error)

        # draw current model
        sinusoid.draw()
        best_model.draw('r-')
        
        plt.waitforbuttonpress(0.01)

        if not plt.fignum_exists(1):
            print("Terminating")
            break
    

        

if __name__ == '__main__':
    main()

