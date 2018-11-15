import cv2 as cv
import numpy as np


if __name__ == "__main__":

    img = cv.imread('key0.png', 1)
    # cv.imshow("start", img)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #THRESHOLD
    ret, binary = cv.threshold(img_gray, 200, 255,cv.THRESH_BINARY_INV)
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    # binary = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    binary_eroded = cv.medianBlur(binary, 1)

    #EDGES DETECTION
    edges = cv.Canny(img_gray, 20, 150)

    lines = cv.HoughLines(edges, 1 , 3*np.pi/180 , 400)
    for i in range(0, 14):
        for rho,theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(binary,(x1,y1),(x2,y2),(0,0,255),2)

    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    # for i in range(200):
    #     for x1,y1,x2,y2 in lines[i]:
    #         cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)



    cv.imshow('edges', img)
    cv.imshow('es', edges)
    cv.imshow('binary', binary)

    #INITIAL BLURRING
    img_gray_blurr = cv.medianBlur(img_gray, 3)


    k = 0
    while k != 27:
        k = cv.waitKey(1)
