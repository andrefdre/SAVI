#!/usr/bin/env python3

import numpy as np
import cv2
import csv
import copy

def main():

    cap = cv2.VideoCapture('./OxfordTownCentre/TownCentreXVID.mp4')
    dataset_location = "./OxfordTownCentre/TownCentre-groundtruth.top"
    data_array= csv.reader(open(dataset_location))

    #Generate random color for each person and store it
    number_person=0
    for row in data_array:
            if len(row) != 12:
                continue
            personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row
            personNumber = int(personNumber)
            if personNumber >= number_person:
                number_person+=1

            colors= np.random.randint(0, high=255,size=(number_person,3),dtype=np.uint8)


    while cap.isOpened():
        ret, frame = cap.read()
        img_gui = copy.deepcopy(frame) #For printing to not lose original image
        frame_height, frame_width = frame.shape[:2]
        currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #img_gui = cv2.resize(img_gui, [frame_width//2, frame_height//2])

        data_array= csv.reader(open(dataset_location))
        for row in data_array:
            if len(row) != 12:
                continue
            personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row
            personNumber = int(personNumber)
            frameNumber = int(frameNumber)
            bodyLeft=int(float(bodyLeft))
            bodyTop=int(float(bodyTop))
            bodyRight=int(float(bodyRight))
            bodyBottom=int(float(bodyBottom))

            if frameNumber!= currentFrame:
                continue
            
            color=colors[personNumber,:]

            cv2.rectangle(img_gui, (bodyLeft,bodyTop), (bodyRight,bodyBottom), (int(color[0]),int(color[1]),int(color[2])), 2, 1)


        cv2.imshow('Initial', img_gui)
        if cv2.waitKey(3) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()