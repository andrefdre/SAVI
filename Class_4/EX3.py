#!/usr/bin/env python3

import numpy as np
import cv2
import csv
import copy
from functions import Detection,Tracker

def main():

    cap = cv2.VideoCapture('./OxfordTownCentre/TownCentreXVID.mp4')
    dataset_location = "./OxfordTownCentre/TownCentre-groundtruth.top"
    data_array= csv.reader(open(dataset_location))

    #Detection Model
    person_detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_fullbody.xml')
    
    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.8

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
    frame_counter = 0
    while cap.isOpened():
         # Step 1: get frame
        ret, image_rgb = cap.read() # get a frame, ret will be true or false if getting succeeds
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        image_gui = copy.deepcopy(image_rgb)
        if ret == False:
            break
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000


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

        # ------------------------------------------
        # Detection of persons 
        # ------------------------------------------
        bboxes = person_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=4, minSize=(20,40))

        detections=[] 
        detection_id = 0 #Id of the detection
        for bbox in bboxes:
            x1, y1, w, h = bbox
            detection = Detection(x1, y1, w, h, image_gray, id=detection_counter)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)
            #cv2.imshow('detection ' + str(detection.id), detection.image  )
            


        #######################################################
        # Compares the new detections to the old ones(tracking)
        #######################################################
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold: # associate detection with tracker 
                    tracker.addDetection(detection)


         # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------
            for detection in detections:
                if not detection.tracker_assigned:
                    tracker = Tracker(detection, id=tracker_counter)
                    tracker_counter += 1
                    trackers.append(tracker)

        # Draw trackers
        for tracker in trackers:
            tracker.draw(image_gui)
        
        print(trackers)

        frame_counter += 1    





        #####################################################
        # Adds new trackers  
        ####################################################

        # for detection in detections:
        #     tracker = Tracking(detection, id=tracker_counter)
        #     color=colors[tracker_counter,:]
        #     detection.draw_rectangle(color)
        #     trackers.append(tracker)
        #     tracker_counter += 1


        cv2.imshow("window_name",image_gui) # show the image
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()