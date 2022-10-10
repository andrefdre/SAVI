#!/usr/bin/env python3

import numpy as np
import cv2 as cv


def main():

    cap = cv.VideoCapture('./OxfordTownCentre/TownCentreXVID.mp4')

    while cap.isOpened():
        ret, frame = cap.read()


        cv.imshow('Initial', frame)
        if cv.waitKey(30) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()