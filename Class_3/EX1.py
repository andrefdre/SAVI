#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import pandas as pd

def get_background(file_path):
    cap = cv.VideoCapture(file_path)
    # we will randomly select 50 frames for the calculating the median
    frame_indices = cap.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
    # we will store the frames in array
    frames = []
    for idx in frame_indices:
        # set the frame id to read that particular frame
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame
 
def main():
    cap = cv.VideoCapture('traffic.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        background_img = get_background('traffic.mp4')
        cv.imshow('frame', background_img)
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()