import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_hist(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def get_mn_max(image):
    mn = [0] * 3
    mx = [0] * 3
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([image], [i], None, [256], [0, 256])
        j = 0
        stop = 1
        for val in histr:
            if(val != 0 and stop):
                mn[i] = j
                stop = 0
            if(val != 0):
                mx[i] = j
            j += 1
    return mn, mx


def get_mn_max_three_percent(image):
    mn = [0] * 3
    mx = [0] * 3
    color = ('b', 'g', 'r')
    three_percent = int(image.size / 3 * 0.03)
    print 'noumber of min_max piels averaged', three_percent
    for i, col in enumerate(color):
        histr = cv.calcHist([image], [i], None, [256], [0, 256])
        total = 0
        stop = 1
        min_count = 0
        max_count = 0
        for h in range(0, len(histr)):
            total += histr[h]
            # print mn
            # if min_count != 0:
                # print mn / min_count

            if total <= three_percent:
                mn[i] += histr[h]*h
                min_count += histr[h]
            elif min_count < three_percent:
                mn[i] += (three_percent - min_count) * h
                min_count += (three_percent - min_count)

        j = -1
        while max_count != three_percent:
            if (max_count+histr[j]) < three_percent:
                mx[i] += histr[j] * (256 + j)
                max_count += histr[j]
            else:
                mx[i] += (three_percent - max_count) * (256 + j)
                max_count += three_percent - max_count
            j -= 1
    print 'Scaling values mn, mx:    ', np.inner(mn, 1.0/three_percent), np.inner(mx, 1.0/three_percent)
    return np.inner(mn, 1.0/three_percent), np.inner(mx, 1.0/three_percent)


def stretch_contrast(image):
    b, g, r = cv.split(image)
    channels = (b, g, r)
    temp = [0]*3
    mn, mx = get_mn_max_three_percent(image)
    i = 0
    for chan in channels:
        dtype = chan.dtype
        chan = chan.astype(float)
        temp[i] = ((chan[:, :]-mn[i])*1.0 / ((mx[i] - mn[i])))*255

        for o in range(0, temp[i].shape[0]):
            for p in range(0, temp[i].shape[1]):
                if temp[i][o, p] > 255:
                    temp[i][o, p] = 255
                elif temp[i][o,p] < 0:
                    temp[i][o, p] = 0

        temp[i] = temp[i].astype(np.uint8)
        
        i += 1
    return cv.merge(temp)


if __name__ == "__main__":

    #TODO: FIX lena color saturation!!!!
    image = cv.imread('../images/lena.png', 1)
    cv.imshow('lenka', image)
    plot_hist(image)
    print get_mn_max(image)
    print get_mn_max_three_percent(image)
    image = stretch_contrast(image)
    plot_hist(image)
    cv.imshow('stretched_lenka', image) 
    cv.waitKey(0)
    cv.imwrite('../images/contrast_stretched_lenna.jpg', image)
    cv.destroyAllWindows()  
 
