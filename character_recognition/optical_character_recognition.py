import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    img = cv.imread('images/key0.png', 1)
    four = cv.imread('images/four.png', 0)
    four = four[110:390, 150:340]
    four = cv.resize(four, (20, 20))
    cv.imshow('dupa', four)
    cv.waitKey(0)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #THRESHOLD
    ret, binary = cv.threshold(img_gray, 200, 255,cv.THRESH_BINARY_INV)
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

    #template
    w, h = four.shape[::-1]
    res = cv.matchTemplate(img_gray, four, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(binary,top_left, bottom_right, 255, 2)
    
    cv.imshow("start", four)
    cv.imshow('binary', binary)

    k = 0
    while k != 27:
        k = cv.waitKey(1)
