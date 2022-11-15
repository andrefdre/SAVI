#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from models import Line


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

    print("Created a figure")
    print(data)
    
    plt.plot(data['xs'],data['ys'],'sk',linewidth=2,markersize=12)

    
    #Define the model
    line=Line(data)
    best_line=Line(data)
    best_error=1E6


    # Execution
    while True:
        #set new values
        line.randomize_params()
        # compute error
        error=line.objectiveFunction()

        if error < best_error:
            best_line.m=line.m
            best_line.b=line.b
            best_error=error
            print("We found a better model")

        print("Error: " ,error)

        # draw current model
        line.draw()
        #best_line.draw('r-')
        plt.draw()

        plt.waitforbuttonpress(0.1)

        if not plt.fignum_exists(1):
            print("Terminating")
            break
    

        

if __name__ == '__main__':
    main()

