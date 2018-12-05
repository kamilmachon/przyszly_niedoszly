import cv2 as cv
import numpy as np


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
    print 'Scaling values mn, mx:    ', np.inner(
        mn, 1.0/three_percent), np.inner(mx, 1.0/three_percent)
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
                elif temp[i][o, p] < 0:
                    temp[i][o, p] = 0

        temp[i] = temp[i].astype(np.uint8)

        i += 1
    return cv.merge(temp)




if __name__ == "__main__":

    image = cv.imread('../images/lena.png')
    # image = stretch_contrast(image)
    cv.imshow('start', image)
    cv.waitKey(0)

    image = np.float32(image) / 255.0

    #gradients
    gx = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=1)
    gy = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=1)
    mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
    # mag = (mag - mag.min()) / (mag.max() - mag.min())*255

    [rows, cols] = image.shape[0], image.shape[1]

    indices = 0
    block_size = 32
    mag_blocks = {}
    angle_blocks = {}
    
    for i in range(0, rows/block_size):
        for j in range(0, cols/block_size):
            mag_blocks[i,j] = mag[i:i+block_size, j:j+block_size, :]
            angle_blocks[i,j] = angle[i:i+block_size, j:j+block_size, :]
            

    # for ang in angle_blocks:
    #     print ang[10]
    #     # exit(0)
    for i in range(0, rows/block_size):
        for j in range(0, cols/block_size):
            for ii in range(0, block_size):
                for jj in range(0, block_size):
                    for k in range(0, 3):
                        # print angle_blocks[i,j][0,0,0]
                        # exit(0)
                        if angle_blocks[i,j][ii, jj,k] > 180:
                            angle_blocks[i,j][ii, jj, k] -= 180

    histograms = {}
    i = 0
    for i in range(0, rows/block_size):
        for j in range(0, cols/block_size):
            print angle_blocks[i,j]

            i += 1 

    cv.imshow('krk', np.uint8(mag)) 
    cv.waitKey(0)
    # print mag.shape , mag



    cv.destroyAllWindows()
