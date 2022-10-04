#!/usr/bin/env python3

import numpy as np
import cv2 as cv

def mean_color(frame,x,y,x1,y1,cars,tic_since_car_count,stamp):
    offset= 40
    cropped_frame = frame[y:y1,x:x1]
    cropped_frame = cv.cvtColor(cropped_frame,cv.COLOR_BGR2GRAY)
    average_color_background = 120
    average_color_row = np.average(cropped_frame, axis=0)
    average_color = np.average(average_color_row, axis=0)
    if  offset > abs(average_color_background - average_color) :
        if stamp-tic_since_car_count > 500:
            cars+=1
            tic_since_car_count=stamp
    return cars, average_color, tic_since_car_count
            

def main():
    cap = cv.VideoCapture('traffic.mp4')
    cars=0
    cars_lane1=0
    cars_lane2=0
    cars_lane3=0
    cars_lane4=0
    recs = [
        {'name': 'rec1' , 'x1' : 150 , 'y1' : 500 , 'x2' : 350 , 'y2' : 600 , 'model_avgColor': 100 , 'tic_since_car_count': 0},
        {'name': 'rec2' , 'x1' : 400 , 'y1' : 500 , 'x2' : 600 , 'y2' : 600 ,'model_avgColor': 100 , 'tic_since_car_count': 0},
        {'name': 'rec3' , 'x1' : 700 , 'y1' : 500 , 'x2' : 900 , 'y2' : 600 ,'model_avgColor': 100 , 'tic_since_car_count': 0},
        {'name': 'rec4' , 'x1' : 1000 , 'y1' : 500 , 'x2' : 1200 , 'y2' : 600 ,'model_avgColor': 100 , 'tic_since_car_count': 0}
    ]
    while cap.isOpened():
        ret, frame = cap.read()
        stamp = float(cap.get(cv.CAP_PROP_POS_MSEC))/1000
        cv.rectangle(frame, (recs[0]['x1'],recs[0]['y1']), (recs[0]['x2'], recs[0]['y2']), (0, 255, 0), 2)
        cv.rectangle(frame, (recs[1]['x1'],recs[1]['y1']), (recs[1]['x2'], recs[1]['y2']), (0, 255, 0), 2)
        cv.rectangle(frame, (recs[2]['x1'],recs[2]['y1']), (recs[2]['x2'], recs[2]['y2']), (0, 255, 0), 2)
        cv.rectangle(frame, (recs[3]['x1'],recs[3]['y1'] ), (recs[3]['x2'], recs[3]['y2']), (0, 255, 0), 2)



        cars_lane1 , avg_color_lane1,recs[0]['tic_since_car_count']= mean_color(frame , recs[0]['x1'] , recs[0]['y1'] , recs[0]['x2'] , recs[0]['y2'] , cars_lane1 ,recs[0]['tic_since_car_count'],stamp)
        cars_lane2 , avg_color_lane2 ,recs[1]['tic_since_car_count']= mean_color(frame , recs[1]['x1'],recs[1]['y1'], recs[1]['x2'], recs[1]['y2'],cars_lane2,recs[1]['tic_since_car_count'],stamp)
        cars_lane3 , avg_color_lane3 ,recs[2]['tic_since_car_count']= mean_color(frame , recs[2]['x1'],recs[2]['y1'], recs[2]['x2'], recs[2]['y2'],cars_lane3,recs[2]['tic_since_car_count'],stamp)
        cars_lane4 , avg_color_lane4 ,recs[3]['tic_since_car_count']= mean_color(frame , recs[3]['x1'],recs[3]['y1'] , recs[3]['x2'], recs[3]['y2'],cars_lane3,recs[3]['tic_since_car_count'],stamp)
        cars=cars_lane1+cars_lane2+cars_lane3+cars_lane4
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv.putText(frame,"No. of cars detected : " + str(cars),(400,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane1),(recs[0]['x1'],recs[0]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane2),(recs[1]['x1'],recs[1]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane3),(recs[2]['x1'],recs[2]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane4),(recs[3]['x1'],recs[3]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.imshow('Initial', frame)
        if cv.waitKey(30) == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()