#TODO:
#detect red, check if round, save its position


# import cv2 as cv
# import numpy as np 


# #actual functions
# ##############################################################################################

# def FindRedDot(image, tracks_array):

#     #transform colorspace to hsv
#     hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     #threshold values to find red
#     lower_red_0 = np.array([0, 30, 30])
#     upper_red_0 = np.array([30, 70, 50])
    
#     lower_red_1 = np.array([150, 100, 100])
#     upper_red_1 = np.array([180, 255, 255])

#     mask0 = cv.inRange(hsv_image, lower_red_0, upper_red_0)
#     mask1 = cv.inRange(hsv_image, lower_red_1, upper_red_1)

#     mask_final = mask0 + mask1

#     # print mask_final

#     cv.imshow("start", mask_final)
        




# #usage part
# ###############################################################################################
# if __name__ == "__main__":
#     cap = cv.VideoCapture(0)
#     tracks_array = [0]

#     while(True):
#         ret, image = cap.read()

#         FindRedDot(image, tracks_array)

    
#         k = cv.waitKey(0)
#         if k == 27:
#             break

#     cv.destroyAllWindows()


# import cv2 as cv
# import numpy as np
# from time import sleep

# def nothing(srsl_nothing):
#     pass

# cap = cv.VideoCapture(0)
# cap.set(3, 4416)
# cap.set(4, 1242)


# uh = 60
# us = 255
# uv = 255

# lh = 34
# ls = 65
# lv = 30

# cv.namedWindow('control panel')
# cv.createTrackbar('Upper H', 'control panel', uh, 180, nothing)
# cv.createTrackbar('Upper S', 'control panel', us, 255, nothing)
# cv.createTrackbar('Upper V', 'control panel', uv, 255, nothing)

# cv.createTrackbar('Lower H', 'control panel', lh, 180, nothing)
# cv.createTrackbar('Lower S', 'control panel', ls, 255, nothing)
# cv.createTrackbar('Lower V', 'control panel', lv, 255, nothing)

# while cap.isOpened():
#     ret, img = cap.read(0)

#     # cv.imshow('start', img)


#     #some filtering
#     #==========================================================================================
#     m_kernel_size = 5
#     bilateral = cv.medianBlur(img, m_kernel_size)
#     # cv.imshow('median', img)

#     # g_kernel_size = (15,15)
#     # img = cv.GaussianBlur(img, g_kernel_size, 2,2)
#     # cv.imshow('gauss', img)

#     # bilateral = cv.bilateralFilter(img, 9, 500, 400) #size, sigma_color, sigma_space
#     # cv.imshow('bilateral', img)


#     #range for allowed colors in hsv color space + hsv convertion
#     #===========================================================================================
#     hsv = cv.cvtColor(bilateral, cv.COLOR_BGR2HSV)
#     #RED
#     # upper_threshold = np.array([80, 255, 255], dtype = np.uint8)
#     # lower_threshold = np.array([35, 0.1*255, 6], dtype = np.uint8)

#     #YELLOW
#     # upper_threshold = np.array([75/2, 255, 255], dtype = np.uint8)
#     # lower_threshold = np.array([30/2, int(0.4*255), 6], dtype = np.uint8)

#     #From trackbars
#     upper_threshold = np.array([cv.getTrackbarPos("Upper H", "control panel"), cv.getTrackbarPos("Upper S", "control panel"), cv.getTrackbarPos("Upper V", "control panel")], dtype = np.uint8)
#     lower_threshold = np.array([cv.getTrackbarPos("Lower H", "control panel"), cv.getTrackbarPos("Lower S", "control panel"), cv.getTrackbarPos("Lower V", "control panel")], dtype = np.uint8)


#     red_only = cv.inRange(hsv, lower_threshold, upper_threshold)
#     cv.imshow('red_only', red_only)


#     k = cv.waitKey(1)
#     if k == 27: #esc key
#         break;

# cv.destroyAllWindows()
