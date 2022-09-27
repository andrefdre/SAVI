#!/usr/bin/env python3

import cv2 as cv
from cv2 import cvtColor
import numpy as np

def main():
    scene = cv.imread('./images/scene.jpg')
    scene_gray = cvtColor(scene,cv.COLOR_BGR2GRAY)
    wally = cv.imread('./images/wally.png',0)
    w, h = wally.shape[::-1]

    res = cv.matchTemplate(scene_gray,wally,eval('cv.TM_CCOEFF'))
    _, _, _, max_loc = cv.minMaxLoc(res)

    print(max_loc)
    cv.rectangle(scene, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255,0), 2)

    cv.imshow('window',scene)
    cv.waitKey(0)

if __name__ == '__main__':
    main()