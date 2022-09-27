#!/usr/bin/env python3

from sys import maxsize
from typing import final
import cv2 as cv
import numpy as np

def main():
    scene = cv.imread('./images/scene.jpg')
    scene_gray = cv.cvtColor(scene,cv.COLOR_BGR2GRAY)
    wally = cv.imread('./images/wally.png')
    h,w,_ = wally.shape[::]
    H_scene , W_scene , _ = scene.shape[::]

    scene_gui = scene[:,:,:]*0

    res = cv.matchTemplate(scene,wally,eval('cv.TM_CCOEFF'))
    _, _, _, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    mask = np.zeros([H_scene,W_scene]).astype(np.uint8)
    cv.rectangle(mask, top_left, bottom_right, 255, -1)
    mask_bool = mask.astype(np.bool)
    scene_gui[mask_bool]=scene[mask_bool]

    negated_mask = np.logical_not(mask_bool)

    image_gray_3 = cv.merge([scene_gray, scene_gray, scene_gray])

    scene_gui[negated_mask] = image_gray_3[negated_mask]


    cv.imshow('Final',scene_gui)
    #cv.imshow('Mask',mask)
    cv.imshow('Scene',scene)
    cv.waitKey(0)

if __name__ == '__main__':
    main()