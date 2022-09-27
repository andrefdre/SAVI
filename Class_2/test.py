#!/usr/bin/env python3

import numpy as np
import cv2 as cv

def main():
    print('hello')
    image= np.ndarray((240,320,3),dtype=np.uint8)
    #When indexing matrices use row,col order
    image = np.random.randint(0,high=255,size=(240,320,3),dtype=np.uint8)

    #set image to gray
    #image = image[:,:,:] + np.random.randint(0,255,dtype=np.uint8)

    cv.imshow('window',image)
    cv.waitKey(0)

#Calls the main function
if __name__ == '__main__':
    main()