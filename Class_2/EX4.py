#!/usr/bin/env python3

from typing import final
import cv2 as cv
import numpy as np

def main():
    scene = cv.imread('./images/scene.jpg')
    final_image = scene.copy()
    scene_gray = cv.cvtColor(scene,cv.COLOR_BGR2GRAY)
    wally = cv.imread('./images/wally.png',0)
    w, h = wally.shape[::-1]
    W, H , _ = scene.shape[::-1]

    scene_gui = scene*0

    res = cv.matchTemplate(scene_gray,wally,eval('cv.TM_CCOEFF'))
    _, _, _, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    mask = np.zeros((H,W)).astype(np.uint8)
    cv.rectangle(mask, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255,0), 2)
    mask_bool = mask.astype(np.bool)
    scene_gui[mask_bool]=scene[mask_bool]

    cv.imshow('window',scene_gui)
    cv.waitKey(0)

if __name__ == '__main__':
    main()