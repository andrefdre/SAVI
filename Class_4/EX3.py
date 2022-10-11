#!/usr/bin/env python3

import numpy as np
import cv2
import csv
import copy
from functions import Detection,Tracking

def main():

    cap = cv2.VideoCapture('./OxfordTownCentre/TownCentreXVID.mp4')
    dataset_location = "./OxfordTownCentre/TownCentre-groundtruth.top"
    data_array= csv.reader(open(dataset_location))

    #Detection Model
    body_detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_fullbody.xml')
    
    #Template Matching
    matching_threshold=0.8

    trackers=[] #Create the trackers memory
    tracker_counter=0
    iou_threshold=0.8

    ####################################################
    #Generate random color for each person and store it#
    ####################################################

    number_person=0
    for row in data_array:
            if len(row) != 12:
                continue
            personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row
            personNumber = int(personNumber)
            if personNumber >= number_person:
                number_person+=1

            colors= np.random.randint(0, high=255,size=(number_person,3),dtype=np.uint8)


    ############################
    # Processes Frame by Frame #
    ############################

    while cap.isOpened():
        ret, frame = cap.read()
        img_gui = copy.deepcopy(frame) #For printing to not lose original image
        frame_height, frame_width = frame.shape[:2]
        current_Frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        img_gui = cv2.resize(img_gui, [frame_width//2, frame_height//2])
        img_gray=cv2.cvtColor(img_gui,cv2.COLOR_BGR2GRAY)


        ##########################################
        # Load dataset to compare results        #
        ##########################################

        # data_array= csv.reader(open(dataset_location))
        # for row in data_array:
        #     if len(row) != 12:
        #         continue
        #     personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row
        #     personNumber = int(personNumber)
        #     frameNumber = int(frameNumber)
        #     bodyLeft=int(float(bodyLeft))
        #     bodyTop=int(float(bodyTop))
        #     bodyRight=int(float(bodyRight))
        #     bodyBottom=int(float(bodyBottom))

        #     if frameNumber!= currentFrame:
        #         continue
            
        #     color=colors[personNumber,:]

        #     cv2.rectangle(img_gui, (bodyLeft,bodyTop), (bodyRight,bodyBottom), (int(color[0]),int(color[1]),int(color[2])), 2, 1)


        #################################################
        # Uses Harr feature of OpenCV to detect people  #
        #################################################

        #Detect the people in the image
        bboxes = body_detector.detectMultiScale(img_gray, scaleFactor=1.2,minNeighbors=4,minSize=(30, 30))
        #bbox is like [x,y,w,h]

        detections=[] 
        detection_id = 0 #Id of the detection
        for bbox in bboxes:
            x,y,w,h = bbox
            detection = Detection(x,y,w,h,img_gui,detection_id)
            detections.append(detection)
            detection.draw_rectangle(color=[255,0,0])
            detection_id+=1
            

        #############################################
        # Create Initial Trackers in first Pixel    #
        #############################################
        if current_Frame == 1:
            for detection in detections:
                tracker = Tracking(detection, id=tracker_counter)
                color=colors[tracker_counter,:]
                detection.draw_rectangle(color)
                trackers.append(tracker)
                tracker_counter += 1


        #######################################################
        # Compares the new detections to the old ones(tracking)
        #######################################################
        for detection in detections:
            for tracker in trackers:
                iou = detection.computeIOU(tracker.detections[-1])
                #print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                res = cv2.matchTemplate(img_gui,detection.image,eval('cv2.TM_CCOEFF_NORMED'))
                print(np.any(res>=matching_threshold))
                print(res)
                if iou > iou_threshold or np.any(res>=matching_threshold):
                    tracker.addDetection(detection)
                    print(detections.index(detection))
                    ##detections.pop(detections.index(detection))
                    color=colors[tracker_counter,:]
                    tracker.draw_rectangle(img_gui,color)

        ##################################################
        #Removed used detections
        ##################################################
            





        #####################################################
        # Adds new trackers  
        ####################################################

        # for detection in detections:
        #     tracker = Tracking(detection, id=tracker_counter)
        #     color=colors[tracker_counter,:]
        #     detection.draw_rectangle(color)
        #     trackers.append(tracker)
        #     tracker_counter += 1


        cv2.imshow('Initial', img_gui)
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()