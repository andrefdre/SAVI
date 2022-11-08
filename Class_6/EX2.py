#!/usr/bin/env python3

import numpy as np
import cv2 
from random import randint


santorini1= cv2.imread("./santorini/1.png")
santorini2= cv2.imread("./santorini/2.png")

santorini1 = cv2.resize(santorini1, (900,900))
santorini2 = cv2.resize(santorini2, (900,900))

gray_santorini1= cv2.cvtColor(santorini1,cv2.COLOR_BGR2GRAY)
gray_santorini2= cv2.cvtColor(santorini2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures=500)
kp_1 , idx = sift.detectAndCompute(gray_santorini1,None)
kp_2 = sift.detect(gray_santorini2,None)

# Using opencv fucntion
#img_1=cv2.drawKeypoints(santorini1,kp_1,None)
#img_2=cv2.drawKeypoints(santorini2,kp_2,None)

for idx,kp in enumerate(kp_1):
    #print('key_poin' + str(idx) + ':' +str(kp))
    #print('x:' +str(kp.pt[0]))
    #print('y:' +str(kp.pt[1]))
    x= int(kp.pt[0])
    y= int(kp.pt[1])
    color = (randint(0,255), randint(0,255), randint(0,255))
    cv2.circle(santorini1,(x,y),20,color,3)

for idx,kp in enumerate(kp_2):
    #print('key_poin' + str(idx) + ':' +str(kp))
    #print('x:' +str(kp.pt[0]))
    #print('y:' +str(kp.pt[1]))
    x= int(kp.pt[0])
    y= int(kp.pt[1])
    color = (randint(0,255), randint(0,255), randint(0,255))
    cv2.circle(santorini2,(x,y),20,color,3)

cv2.imshow("santorini1",santorini1)
cv2.imshow("santorini2",santorini2)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()