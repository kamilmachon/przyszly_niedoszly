import cv2 as cv 
import numpy as np 



if __name__ == "__main__":
    image = cv.imread('../images/shapes.png')

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 137, 255, cv.THRESH_BINARY_INV)

    gray, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=False)
    # cv.drawContours(image, contours, -1, (0, 255, 0), 3)


    M = {}
    i = 0
    for cnt in contours:
            #calculate mass center
            try:
                M = cv.moments(cnt)
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                cv.circle(image, (x,y), 5, (0,0,255), 10)
                print 'Center: ', x, y 
                print 'Area:', cv.contourArea(cnt)
                (x,y), theta , = cv.minEnclosingCircle(cnt)
                print 'Oritntation', theta 
                area = cv.contourArea(cnt)
                hull = cv.convexHull(cnt)
                hull_area = cv.contourArea(hull)
                solidity = float(area)/hull_area
                print 'Solidity', solidity
                print 'rownowazny promien', np.sqrt(4*area/np.pi)
                x, y, w, h = cv.boundingRect(cnt)
                aspect_ratio = float(w)/h
                print 'aspect ratio', aspect_ratio
            except:
                pass

            #plot line
            try:
                rows, cols = image.shape[:2]
                [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                print (cols-1, righty), (0, lefty)
                cv.line(image, (cols-1, righty), (0, lefty), (255, 0, 0), 2)
            except:
                pass





            i += 1

    #         area = cv.contourArea(cnt)
    #         (x, y), radius = cv.minEnclosingCircle(cnt)
    #         density = np.pi*(radius**2) / area
    #         if min_radius < radius < max_radius:
    #             if density < 3:
    #                 print radius
    #             # print (int(x), int(y)), int(radius)
    #                 center_histry_x.insert(0, int(x))
    #                 center_histry_y.insert(0, int(y))
    #                 try:
    #                     cv.circle(image, (int(x), int(y)),
    #                               int(radius), (0, 255, 0), 1)
    #                     cv.circle(red, (int(x), int(y)), int(radius), 120, 1)
    #                 except:
    #                     print 'failed to draw'

    #                 draw = True
    #             else:
                    # print density



    cv.imshow('start', image)
    # cv.imshow('tr', thresh)
    cv.waitKey(0)

    cv.destroyAllWindows()
    print 'did'
