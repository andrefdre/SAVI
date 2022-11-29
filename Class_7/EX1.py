#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.grid(True)
    plt.xlabel('x')
    plt.xlabel('y')

    print("Created a figure")

    pts={
        'xs':[],
        'ys':[]
    }
    file = open('pts.pk1', 'wb')
    while(True):
        plt.plot(pts['xs'],pts['ys'],'rx',linewidth=2,markersize=12)
        pt=plt.ginput(1)
        if not pt:
            print("Terminated")
            break

        pts['xs'].append(pt[0][0])
        pts['ys'].append(pt[0][1])
        print(str(pts))
        
    pickle.dump(pts,file)
    file.close()

if __name__ == '__main__':
    main()

