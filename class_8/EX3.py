#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from scipy.optimize import least_squares
from models_1 import ImageMosaic
import cv2 
from copy import deepcopy


def main():
    q_image= cv2.imread("./marvao/2.png")
    q_gui=deepcopy(q_image)
    t_image= cv2.imread("./marvao/1.png")
    t_gui=deepcopy(t_image)


if __name__ == '__main__':
    main()