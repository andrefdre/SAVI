#!/usr/bin/env python3

import numpy as np
import cv2 as cv

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy
 
def main():
    cap = cv.VideoCapture('traffic.mp4')
    largura_min = 21
    altura_min = 21
    offset = 6
    pos_linha = 550
    detec = []
    carros = 0
    #background_img = get_background('traffic.mp4')
    background = cv.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(frame_gray,(3,3),5)
        img_sub = background.apply(blur)
        ret,final = cv.threshold(img_sub,100,255,cv.THRESH_BINARY)
    
        kernel = np.ones((5,5),np.uint8)
        final = cv.erode(final,kernel,iterations=2)
        final = cv.medianBlur(final,3)
        # Fill any small holes
        final = cv.morphologyEx(final, cv.MORPH_CLOSE, kernel)
        final = cv.dilate(final,kernel,iterations=4)

        contours, hierarchy = cv.findContours(final, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cv.line(frame, (25, pos_linha), (1200, pos_linha), (176, 130, 39), 2)

        for c in contours:
            (x, y, w, h) = cv.boundingRect(c)
            contour_valid = (w >= largura_min) and (h >= altura_min)
            if not contour_valid:
                continue

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            centro = pega_centro(x, y, w, h)
            detec.append(centro)
            cv.circle(frame, centro, 4, (0, 0, 255), -1)

            for (x, y) in detec:
                if (y < (pos_linha + offset)) and (y > (pos_linha-offset)):
                    carros += 1
                    cv.line(frame, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
                    detec.remove((x, y))
                    print("No. of cars detected : " + str(carros))

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.putText(frame,"No. of cars detected : " + str(carros),(400,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow('frame', final)
        cv.imshow('Initial', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()