#!/usr/bin/env python3

import numpy as np
import cv2 
from random import randint
from copy import deepcopy

#Podia chamar q e t as imagens um de query e uma target
q_castle1= cv2.imread("./castle/1.png")
q_gui=deepcopy(q_castle1)
t_castle2= cv2.imread("./castle/2.png")
t_gui=deepcopy(t_castle2)
#q_gui = cv2.resize(q_gui, (500,500))
#_gui = cv2.resize(t_gui, (500,500))
gray_castle1= cv2.cvtColor(q_gui,cv2.COLOR_BGR2GRAY)
gray_castle2= cv2.cvtColor(t_gui,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures=200)
q_kp , q_des = sift.detectAndCompute(gray_castle1,None)
t_kp , t_des = sift.detectAndCompute(gray_castle2,None)

index_pararms = dict(algorithm=1 , trees=15)
search_params = dict(checks=50)
flann=cv2.FlannBasedMatcher(index_pararms,search_params)
best_two_matches = flann.knnMatch(q_des,t_des,k=2)

#Create a list with only the best matches and David Lowe's ratio test to comute the uniqueness of a match
matches=[]
for best_two_match in best_two_matches:
    best_match = best_two_match[0]
    second_match = best_two_match[1]
    best_match_distance=best_match.distance
    second_match_distance=second_match.distance
    # David Lowe's test
    if best_match_distance < 0.3 * second_match_distance:
        matches.append(best_match)

# Using opencv fucntion
#q_gui=cv2.drawKeypoints(q_gui,kp_1,None,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
#t_gui=cv2.drawKeypoints(t_gui,kp_2,None,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

for idx,kp in enumerate(q_kp):
    #print('key_poin' + str(idx) + ':' +str(kp))
    #print('x:' +str(kp.pt[0]))
    #print('y:' +str(kp.pt[1]))
    x= int(kp.pt[0])
    y= int(kp.pt[1])
    color = (randint(0,255), randint(0,255), randint(0,255))
    cv2.circle(q_gui,(x,y),20,color,3)

for idx,kp in enumerate(t_kp):
    #print('key_poin' + str(idx) + ':' +str(kp))
    #print('x:' +str(kp.pt[0]))
    #print('y:' +str(kp.pt[1]))
    x= int(kp.pt[0])
    y= int(kp.pt[1])
    color = (randint(0,255), randint(0,255), randint(0,255))
    cv2.circle(t_gui,(x,y),20,color,3)


#Shows the matches image
matches_image = cv2.drawMatches(q_gui,q_kp,t_gui,t_kp,matches,None)

cv2.namedWindow("castle1",cv2.WINDOW_NORMAL)
cv2.imshow("castle1",q_gui)
cv2.resizeWindow("castle1",600,400)
cv2.namedWindow("castle2",cv2.WINDOW_NORMAL)
cv2.resizeWindow("castle2",600,400)
cv2.imshow("castle2",t_gui)
cv2.namedWindow("Matches",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Matches",600,400)
cv2.imshow("Matches",matches_image)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()