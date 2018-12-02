import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def plot_hist(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def normalize_max_channel(image):
    [g,b,r] = cv.split(image)

    [g_mean, b_mean, r_mean] = [np.mean(g), np.mean(b), np.mean(r)]
    means =  [g_mean, b_mean, r_mean]
    global_mean = np.mean(image)
    scale = max(means) / global_mean
    r = r/scale
    r = r.astype(np.uint8)
    channels = [g,b,r]
    return cv.merge(channels)
    
def normalize_all_channels(image):
    [g, b, r] = cv.split(image)
    global_mean = np.mean(image)
    print 'global: ', global_mean, "R: ", np.mean(r), 'G: ', np.mean(g), 'B: ', np.mean(b)

    g = g.astype(float)
    g = g * global_mean / np.mean(g)

    b = b.astype(float)
    b = b * global_mean / np.mean(b)

    r = r.astype(float)
    r = r * global_mean / np.mean(r)

    channels = [g, b, r]


    for chan in channels:
        for i in range(0, chan.shape[0]):
            for j in range(0, chan.shape[1]):
                if chan[i,j] > 255:
                    chan[i,j] = 255
                elif chan[i,j] < 0:
                    chan[i,j] = 0

    image = cv.merge(channels)

    return image.astype(np.uint8)

if __name__ == "__main__":
    image = cv.imread('../images/lena.png', 1)
    cv.imshow('start', image)
    normalize_max_channel = normalize_max_channel(image)
    normalize_all_channels = normalize_all_channels(image)

    cv.imshow('normalize_max_channel', normalize_max_channel)
    cv.imshow('normalize_all_channels', normalize_all_channels)
    cv.imwrite('../images/all_channels_normalized_lena.jpg',normalize_all_channels )
    cv.imwrite('../images/max_channel_normalized_lena.jpg',normalize_max_channel )
    cv.waitKey(0)
    cv.destroyAllWindows()

    plot_hist(image)
    plot_hist(normalize_max_channel)
    plot_hist(normalize_all_channels)
