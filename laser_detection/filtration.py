import cv2 as cv 
import numpy as np 



image = cv.imread('../images/lena.png')
cv.imshow('start', image)
image = cv.bilateralFilter(image, 15, 100, 3)
cv.imshow('done', image)
cv.waitKey(0)
cv.destroyAllWindows()
