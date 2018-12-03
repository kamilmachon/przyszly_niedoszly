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


def retinex_adjust(image):
    image = image.astype(float)
    [B,G,R] = cv.split(image)
    R = R * G.max() / R.max()
    B = B * G.max() / B.max()
    
    coefficient = np.linalg.solve(np.array([[(R**2).sum() , R.sum()], [(R**2).max(), R.max()]]), np.array([G.sum(), G.max()]))
    print coefficient, np.array([[(R**2).sum(), R.sum()], [(R**2).max(), R.max()]]),  np.array([G.sum(), G.max()])
    R = (R ** 2) * coefficient[0] + R*coefficient[1]

    coefficient = np.linalg.solve(np.array([[(B ** 2).sum(), B.sum()], [(B ** 2).max(), B.max()]]), np.array([G.sum(), G.max()]))
    B = (B ** 2) * coefficient[0] + B*coefficient[1]

    channels = (B,G,R)
    image = cv.merge(channels)
    image = np.minimum(image, 255)
    image = np.maximum(image, 0)


    return image.astype(np.uint8)



if __name__ == "__main__":

    image = cv.imread('../images/dark_room.jpg')
    plot_hist(image)
    cv.imshow('lenka', image)
    image = retinex_adjust(image)

    cv.imshow('lenka2', image)
    # cv.imwrite('retinex_lena.jpg', image)
    plot_hist(image)
    cv.waitKey(0)
    cv.destroyAllWindows()
