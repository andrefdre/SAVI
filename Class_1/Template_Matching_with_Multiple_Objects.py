#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('mario.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
template = cv.imread('mario_coin.png',0)

w, h = template.shape[::-1]


res = cv.matchTemplate(img_gray,template,eval('cv.TM_CCOEFF_NORMED'))
threshold = 0.8
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv.imshow('res.png',img)
cv.waitKey() 

