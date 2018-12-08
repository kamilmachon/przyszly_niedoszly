import cv2 as cv
import numpy as np 
from retinex import retinex_adjust
from contrast_stretching import stretch_contrast


def nothing(nothing):
    pass


uh = 10
us = 80
uv = 60

lh = 0
ls = 100
lv = 100

cv.namedWindow('control panel1')
cv.createTrackbar('Upper H', 'control panel1', uh, 180, nothing)
cv.createTrackbar('Upper S', 'control panel1', us, 255, nothing)
cv.createTrackbar('Upper V', 'control panel1', uv, 255, nothing)

cv.createTrackbar('Lower H', 'control panel1', lh, 180, nothing)
cv.createTrackbar('Lower S', 'control panel1', ls, 255, nothing)
cv.createTrackbar('Lower V', 'control panel1', lv, 255, nothing)

uh = 180
us = 255
uv = 255

lh = 170
ls = 100
lv = 100

cv.namedWindow('control panel2')
cv.createTrackbar('Upper H', 'control panel2', uh, 180, nothing)
cv.createTrackbar('Upper S', 'control panel2', us, 255, nothing)
cv.createTrackbar('Upper V', 'control panel2', uv, 255, nothing)

cv.createTrackbar('Lower H', 'control panel2', lh, 180, nothing)
cv.createTrackbar('Lower S', 'control panel2', ls, 255, nothing)
cv.createTrackbar('Lower V', 'control panel2', lv, 255, nothing)



if __name__ == "__main__":
    cap = cv.VideoCapture(1)
    cap.set(3, 4416/2)
    cap.set(4, 1242/2)
    elems_to_keep = 100
    center_histry_x = [0] * elems_to_keep
    center_histry_y = [0] * elems_to_keep




    while cap.isOpened() :
        draw = False
        ret, image = cap.read()
        image = image[:,image.shape[1]/2:image.shape[1]]
        # retinex = retinex_adjust(image)
        # contrast = stretch_contrast(image)
        # cv.imshow('retinex', retinex)
        # cv.imshow('contrast', contrast)
        image = cv.bilateralFilter(image, 9, 2000, 900)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        upper_threshold = np.array([cv.getTrackbarPos("Upper H", "control panel1"), cv.getTrackbarPos(
            "Upper S", "control panel1"), cv.getTrackbarPos("Upper V", "control panel1")], dtype=np.uint8)
        lower_threshold = np.array([cv.getTrackbarPos("Lower H", "control panel1"), cv.getTrackbarPos(
            "Lower S", "control panel1"), cv.getTrackbarPos("Lower V", "control panel1")], dtype=np.uint8)
        red1 = cv.inRange(hsv, lower_threshold, upper_threshold)

        upper_threshold = np.array([cv.getTrackbarPos("Upper H", "control panel2"), cv.getTrackbarPos(
            "Upper S", "control panel2"), cv.getTrackbarPos("Upper V", "control panel2")], dtype=np.uint8)
        lower_threshold = np.array([cv.getTrackbarPos("Lower H", "control panel2"), cv.getTrackbarPos(
            "Lower S", "control panel2"), cv.getTrackbarPos("Lower V", "control panel2")], dtype=np.uint8)
        red2 = cv.inRange(hsv, lower_threshold, upper_threshold)

        red = red1 + red2

        kernel = np.ones([5, 5])
        # red = cv.erode(red, kernel, iterations=1)
        red = cv.dilate(red, kernel, iterations=1)

    
        #contours

        im2, contours, hierarchy = cv.findContours(red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=False)

        min_area_size = 0
        max_area_size = 9999999
        min_radius = 9
        max_radius = 50

        i = 0
        fit_old = 999
        index = 999
        center = [0, 0]



        for cnt in contours:
            area = cv.contourArea(cnt)
            (x,y),radius = cv.minEnclosingCircle(cnt)
            density = np.pi*(radius**2) / area
            if min_radius < radius < max_radius:
                if density < 3:
                    print radius
                # print (int(x), int(y)), int(radius)
                    center_histry_x.insert( 0, int(x))
                    center_histry_y.insert( 0, int(y))
                    try:
                        cv.circle(image, (int(x),int(y)), int(radius), (0,255,0), 1)
                        cv.circle(red, (int(x),int(y)), int(radius), 120, 1)
                    except:
                        print 'failed to draw'

                    draw = True
                else:
                    print density

            # print x,y,radius
                # try:
                #     ellipse = cv.fitEllipse(cnt)
                #     # print 'HERE',  ellipse[1][0], ellipse[1][1]
                #     fit = float(np.abs(ellipse[1][0] - ellipse[1][1]))
                #     if fit < fit_old:
                #         fit_old = fit
                #         index = i
                #         center = (ellipse[0][0], ellipse[0][1])
                #         # print 'fit:', fit
                #     cv.drawContours(image, cnt, 0,  (255, 0, 0), 3)
                # except:
                #     pass
                # print 'fit:'
                # cv.ellipse(img,ellipse,(0,255,0),2)
                # print center
        
        history = red
        for i in range(elems_to_keep-1):
            # print center_histry_x[i], center_histry_y[i]
            if center_histry_x[i+1] > 0 and center_histry_y[i] > 0:
                # print 'value', int((elems_to_keep - float(i))/elems_to_keep * 255)
                # cv.circle(history, (center_histry_x[i], center_histry_y[i]), 5, int((elems_to_keep - float(i))/elems_to_keep * 255), 6)
                cv.line(image, (center_histry_x[i], center_histry_y[i]), (center_histry_x[i+1], center_histry_y[i+1]), (0, 0, int((elems_to_keep - float(i))/elems_to_keep * 255) ), 2)

        if draw == True:
            center_histry_x = center_histry_x[0:elems_to_keep]
            center_histry_y = center_histry_y[0:elems_to_keep]
        # print center_histry



        cv.imshow('start', image)
        cv.imshow('red', red)
        key = cv.waitKey(1)
        if key == 27:
            break
    
    cv.destroyAllWindows()
