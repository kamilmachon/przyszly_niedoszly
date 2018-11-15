import cv2 as cv
import numpy as np


if __name__ == "__main__":

    img = cv.imread('key0.png', 1)
    cv.imshow("start", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #THRESHOLDING
    ret, thresh = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imshow("thresholded", thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #REMOVE NOISE
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 1)    

    #BACKGROUND
    sure_bg = cv.dilate(opening,kernel,iterations=0)
    cv.imshow("dilate", sure_bg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    #FOREGROUND
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    cv.imshow("foreground", sure_fg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #UNKNOWN
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    cv.imshow("unknown", unknown)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    cv.imshow("markers", markers)
    cv.waitKey(0)
    cv.destroyAllWindows()

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    cv.imshow("markers_final", markers)
    cv.waitKey(0)
    cv.destroyAllWindows()
    