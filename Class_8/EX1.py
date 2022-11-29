#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from scipy.optimize import least_squares
from models_1 import ImageMosaic
import cv2 
from copy import deepcopy


def main():
    q_warped= cv2.imread("./machu_pichu/query_warped.png")
    q_gui=deepcopy(q_warped)
    q_stich= cv2.imread("./machu_pichu/query_stitched.png")
    q_stich_gui=deepcopy(q_stich)
    target= cv2.imread("./machu_pichu/target.png")
    t_gui=deepcopy(target)

    image_mosaic = ImageMosaic(q_gui,t_gui)
    x0=[image_mosaic.q_scale,image_mosaic.q_bias,image_mosaic.t_scale,image_mosaic.t_bias]
    result= least_squares(image_mosaic.objectiveFunction,x0,verbose=2)

    image_mosaic.draw()
    cv2.waitKey(0)


if __name__ == '__main__':
    main()