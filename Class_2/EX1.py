#!/usr/bin/env python3

import math
import cv2 as cv
import numpy as np

def main():
    img = cv.imread('./images/lake.jpg')
    res = img.copy()
    h, w , _ = img.shape[::]
    index = int(w/2)
    # _ is trash variable

    for i in np.arange(1,0.2,-0.01):
        res[:,index:w,:] = (img[:,index:w,:] *i).astype(np.uint8)
        cv.imshow('window',res)
        cv.waitKey(10)
    
    cv.waitKey(0)


if __name__ == '__main__':
    main()