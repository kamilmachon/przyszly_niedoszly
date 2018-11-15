import cv2 as cv
import numpy as np

# src_points = [[115,406], [883, 389], [5, 1319], [1043, 1289] ]
src_points = []
scale = 30
dst_points = np.array([[0,0], [20 * scale,0], [0,24*scale], [20*scale, 24*scale]])


def on_mouse_click(event, x, y, flag, param):
    global src_points
    
    if event == cv.EVENT_LBUTTONDOWN:
        src_points += [[x,y]]
        # print type(src_points)
        # print src_points
    
def process_src_points(src_points):

    a = src_points[2][1] - src_points[0][1]
    b = src_points[3][0] - src_points[2][0]
    c = np.sqrt( (src_points[3][0] - src_points[0][0])**2 + (src_points[3][1] - src_points[0][1])**2)
    alfa = np.arccos( (a**2 + b**2 - c**2)/(2*a*b) )
    calculated_a = a / np.sin(alfa)

    print src_points
    print alfa
    print calculated_a

def perspective_correction(image, src_points, dst_points):

    process_src_points(src_points)

    src_points = np.array(src_points)
    h, status = cv.findHomography(src_points, dst_points)
    dst_image = cv.warpPerspective(image, h, (600, int(600*1.2)) )
    for center in src_points:
        cv.circle(image, tuple(center), 10, (0, 0, 255), 2)
    print h
    cv.imshow("final", dst_image)
    cv.imshow("marked", image)
    cv.waitKey(0)
    cv.imwrite("../images/feynman_corrected.jpg", dst_image)
    cv.imwrite("../images/feynman_marked.jpg", image)
    exit(0)


if __name__ == "__main__":
    
    cv.namedWindow("before_correction")
    cv.setMouseCallback("before_correction", on_mouse_click)
    image = cv.imread("../images/feynman.jpg", 1)

    while True:
        cv.imshow("before_correction", image)
        key = cv.waitKey(1)

        if len(src_points) == 4:
            perspective_correction(image, src_points, dst_points)
            break

        if key == ord("q"):
            break


    cv.destroyAllWindows()