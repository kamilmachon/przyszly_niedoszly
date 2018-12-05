import cv2 as cv
import numpy as np 

block_size = 32


if __name__ == "__main__":
    image = cv.imread("../images/retinex_lena.jpg")
    [rows, cols] = image.shape[0], image.shape[1]
    
    indices = 0
    #for square pictures only
    while rows > indices:

        cv.line(image, (0, indices), (rows, indices), (255, 0, 0), 2)
        cv.line(image, (indices, 0), (indices, rows), (255, 0, 0), 2)
        # print 'line points',  (0, indices), (rows, indices)
        indices += block_size

    cv.imshow("splited_lena", image)
    cv.waitKey(0)

cv.destroyAllWindows()
