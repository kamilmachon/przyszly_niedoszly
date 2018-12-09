import struct
import numpy as np
import cv2 as cv

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def prepare_data():
    images = read_idx('emnist-letters-test-images-idx3-ubyte')
    labels = read_idx('emnist-letters-test-labels-idx1-ubyte')
    rotate = cv.getRotationMatrix2D((14,14),-90,1)
    i = 0
    for image in images:
        images[i] = cv.warpAffine(image,rotate,(28,28))
        images[i] = cv.flip(image, 1)
        i += 1
    return images, labels

if __name__ == "__main__":
    
    images, labels = prepare_data()

    i = 0
    while(True):
        cv.imshow('dd', images[i])
        print labels[i]
        k = cv.waitKey(0)
        i += 1
        if(k == 27):
            break