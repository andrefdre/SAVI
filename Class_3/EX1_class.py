#!/usr/bin/env python3

import numpy as np
import cv2 as cv

def detect_car(frame,x,y,x1,y1,cars,tic_since_car_count,stamp , average_color_background , color_car):
    offset= 20
    cropped_frame = frame[y:y1,x:x1]
    cropped_frame_Gray = cv.cvtColor(cropped_frame,cv.COLOR_BGR2GRAY)
    average_color_row = np.average(cropped_frame_Gray, axis=0)
    average_color = np.average(average_color_row, axis=0)
    if  offset < abs(average_color_background - average_color):
        if stamp-tic_since_car_count > 1:
            cars=cars+1
            tic_since_car_count=stamp
            color_car = findCar_color(frame[y-30:y1,x:x1])
            print('Car detected color diff: ' + str(abs(average_color_background - average_color)))
    return cars, average_color, tic_since_car_count , color_car


def findCar_color(frame):
    cropped_frame=frame[0:10,80:100]
    cv.imshow("teste",cropped_frame)
    cv.rectangle(frame, (80,0), (100, 10), (0, 255, 0), 2)
    average_color_row = np.median(cropped_frame, axis=0)
    average_color = np.median(average_color_row, axis=0)
    print(average_color)
    return  average_color
            


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def main():
    cap = cv.VideoCapture('traffic.mp4')
    cars=0
    recs = [
        {'name': 'rec1' , 'x1' : 150 , 'y1' : 500 , 'x2' : 350 , 'y2' : 600 , 'model_avgColor': 100 , 'tic_since_car_count': 0 , 'AverageBackground_compare' : 115},
        {'name': 'rec2' , 'x1' : 400 , 'y1' : 500 , 'x2' : 600 , 'y2' : 600 ,'model_avgColor': 100 , 'tic_since_car_count': 0 , 'AverageBackground_compare' : 110},
        {'name': 'rec3' , 'x1' : 700 , 'y1' : 500 , 'x2' : 900 , 'y2' : 600 ,'model_avgColor': 100 , 'tic_since_car_count': 0 , 'AverageBackground_compare' : 128},
        {'name': 'rec4' , 'x1' : 950 , 'y1' : 500 , 'x2' : 1150 , 'y2' : 600 ,'model_avgColor': 100 , 'tic_since_car_count': 0 , 'AverageBackground_compare' : 118}
    ]
    avg_color_lane0_bgr=(0,0,0)
    avg_color_lane1_bgr=(0,0,0)
    avg_color_lane2_bgr=(0,0,0)
    avg_color_lane3_bgr=(0,0,0)
    while cap.isOpened():
        ret, frame = cap.read()
        stamp = float(cap.get(cv.CAP_PROP_POS_MSEC))/1000

        cv.rectangle(frame, (recs[0]['x1'],recs[0]['y1']), (recs[0]['x2'], recs[0]['y2']), (0, 255, 0), 2)
        cv.rectangle(frame, (recs[1]['x1'],recs[1]['y1']), (recs[1]['x2'], recs[1]['y2']), (0, 255, 0), 2)
        cv.rectangle(frame, (recs[2]['x1'],recs[2]['y1']), (recs[2]['x2'], recs[2]['y2']), (0, 255, 0), 2)
        cv.rectangle(frame, (recs[3]['x1'],recs[3]['y1'] ), (recs[3]['x2'], recs[3]['y2']), (0, 255, 0), 2)



        cars , avg_color_lane1,recs[0]['tic_since_car_count'] , avg_color_lane0_bgr= detect_car(frame , recs[0]['x1'] , recs[0]['y1'] , recs[0]['x2'] , recs[0]['y2'] , cars ,recs[0]['tic_since_car_count'],stamp , recs[0]['AverageBackground_compare'] , avg_color_lane0_bgr)
        cars , avg_color_lane2 ,recs[1]['tic_since_car_count'] , avg_color_lane1_bgr=detect_car(frame , recs[1]['x1'],recs[1]['y1'], recs[1]['x2'], recs[1]['y2'],cars,recs[1]['tic_since_car_count'],stamp , recs[1]['AverageBackground_compare'] , avg_color_lane1_bgr)
        cars , avg_color_lane3 ,recs[2]['tic_since_car_count'] , avg_color_lane2_bgr=detect_car(frame , recs[2]['x1'],recs[2]['y1'], recs[2]['x2'], recs[2]['y2'],cars,recs[2]['tic_since_car_count'],stamp , recs[2]['AverageBackground_compare'] , avg_color_lane2_bgr)
        cars , avg_color_lane4 ,recs[3]['tic_since_car_count'] , avg_color_lane3_bgr=detect_car(frame , recs[3]['x1'],recs[3]['y1'] , recs[3]['x2'], recs[3]['y2'],cars,recs[3]['tic_since_car_count'],stamp , recs[3]['AverageBackground_compare'] , avg_color_lane3_bgr)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv.putText(frame,"No. of cars detected : " + str(cars),(400,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame,"Stamp : " + str(stamp),(10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane1),(recs[0]['x1'],recs[0]['y1']-50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane2),(recs[1]['x1'],recs[1]['y1']-50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane3),(recs[2]['x1'],recs[2]['y1']-50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Color : " + str(avg_color_lane4),(recs[3]['x1'],recs[3]['y1']-50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        cv.putText(frame,"Color_BGR :" + str(avg_color_lane0_bgr),(recs[0]['x1'],recs[0]['y1']-30), cv.FONT_HERSHEY_SIMPLEX, 0.3, avg_color_lane0_bgr, 1, cv.LINE_AA)
        cv.putText(frame,"Color_BGR :" + str(avg_color_lane1_bgr),(recs[1]['x1'],recs[1]['y1']-30), cv.FONT_HERSHEY_SIMPLEX, 0.3, avg_color_lane1_bgr, 1, cv.LINE_AA)
        cv.putText(frame,"Color_BGR : " + str(avg_color_lane2_bgr),(recs[2]['x1'],recs[2]['y1']-30), cv.FONT_HERSHEY_SIMPLEX, 0.3, avg_color_lane2_bgr, 1, cv.LINE_AA)
        cv.putText(frame,"Color_BGR : " + str(avg_color_lane3_bgr),(recs[3]['x1'],recs[3]['y1']-30), cv.FONT_HERSHEY_SIMPLEX, 0.3, avg_color_lane3_bgr, 1, cv.LINE_AA)

        cv.putText(frame,"Tic_lastCar : " + str(recs[0]['tic_since_car_count']),(recs[0]['x1'],recs[0]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Tic_lastCar : " + str(recs[1]['tic_since_car_count']),(recs[1]['x1'],recs[1]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Tic_lastCar : " + str(recs[2]['tic_since_car_count']),(recs[2]['x1'],recs[2]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame,"Tic_lastCar : " + str(recs[3]['tic_since_car_count']),(recs[3]['x1'],recs[3]['y1']-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                

        cv.imshow('Initial', frame)
        if cv.waitKey(30) == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()