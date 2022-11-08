#!/usr/bin/env python3

import numpy as np
import cv2 
from random import randint
from copy import deepcopy

MIN_MATCH_COUNT = 10


q_machu_pichu= cv2.imread("./machu_pichu/1.png")
q_gui=deepcopy(q_machu_pichu)
q_stich_gui=deepcopy(q_machu_pichu)
t_machu_pichu= cv2.imread("./machu_pichu/2.png")
t_gui=deepcopy(t_machu_pichu)


gray_q_gui= cv2.cvtColor(q_gui,cv2.COLOR_BGR2GRAY)
gray_t_gui= cv2.cvtColor(t_gui,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures=200)
q_kp , q_des = sift.detectAndCompute(gray_q_gui,None)
t_kp , t_des = sift.detectAndCompute(gray_t_gui,None)

index_pararms = dict(algorithm=1 , trees=15)
search_params = dict(checks=50)
flann=cv2.FlannBasedMatcher(index_pararms,search_params)
best_two_matches = flann.knnMatch(q_des,t_des,k=2)

#Create a list with only the best matches and David Lowe's ratio test to comute the uniqueness of a match
pts=[]
matrix=[]
matches=[]
for best_two_match in best_two_matches:
    best_match = best_two_match[0]
    m = best_two_match[0]
    second_match = best_two_match[1]
    best_match_distance=best_match.distance
    second_match_distance=second_match.distance
    # David Lowe's test
    if best_match_distance < 0.3 * second_match_distance:
        matches.append(best_match)

    if len(matches)>MIN_MATCH_COUNT:
        query_pts = np.float32([q_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) 
        train_pts = np.float32([t_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) 
        matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w, _ = t_gui.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    else:
        #print( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
        matchesMask = None

dst = cv2.perspectiveTransform(pts,matrix)
h,w, _ = q_stich_gui.shape
q_image_warped=cv2.warpPerspective(t_gui,matrix,(w,h))
q_stich_gui= cv2.polylines(q_stich_gui,[np.int32(dst)],True,255,3, cv2.LINE_AA)
overlap_mask=q_image_warped>0
q_stich_gui[overlap_mask]=q_image_warped[overlap_mask]
cv2.namedWindow("Homography",cv2.WINDOW_NORMAL)
cv2.imshow("Homography", q_stich_gui) 
cv2.resizeWindow("Homography",600,400)


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