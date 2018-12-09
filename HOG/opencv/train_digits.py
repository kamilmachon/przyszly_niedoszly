#!/usr/bin/env python

import cv2
import struct
import numpy as np

SZ = 28
CLASS_N = 26

# local modules
from common import clock, mosaic

# def split2d(img, cell_size, flatten=True):
#     h, w = img.shape[:2]
#     sx, sy = cell_size
#     cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
#     cells = np.array(cells)
#     if flatten:
#         cells = cells.reshape(-1, sy, sx)
#     return cells

# def load_digits(fn):
#     digits_img = cv2.imread(fn, 0)
#     digits = split2d(digits_img, (SZ, SZ))
#     labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
#     return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def svmInit(C=12.5, gamma=0.50625):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  
  return model

def svmTrain(model, samples, responses):
#   print type(responses[0])
  responses = np.int64(responses)
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, digits, samples, labels):
    predictions = svmPredict(model, samples)
    print predictions

    accuracy = (labels == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy*100))

    confusion = np.zeros((CLASS_N+1, CLASS_N+1), np.int32)
    print predictions.shape
    print labels.shape
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    vis = []
    # i = 0
    for img, flag in zip(digits, predictions == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        # print str(unichr(int(labels[i]) + 96))
        # print str(unichr(int(predictions[i]) + 96))
        # cv2.imshow('dupaia', img)
        # cv2.waitKey(0)
        # i += 1
        vis.append(img)
    return mosaic(25, vis)


def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0


def get_hog() : 
    winSize = (28,28)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

#MY OWN
def preprocess(hog):
    image = cv2.imread('four.png', 0)
    ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    image = cv2.resize(image, (28,28))
    image = hog.compute(image)
    image = np.swapaxes(image, 1, 0)
    return image

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def prepare_data():
    images = read_idx('emnist-letters-test-images-idx3-ubyte')
    labels = read_idx('emnist-letters-test-labels-idx1-ubyte')
    rotate = cv2.getRotationMatrix2D((14,14),-90,1)
    i = 0
    for image in images:
        images[i] = cv2.warpAffine(image,rotate,(28,28))
        images[i] = cv2.flip(images[i], 1)
        i += 1
    return images, labels


if __name__ == '__main__':

    print('Loading digits from digits.png ... ')
    # Load data.
    digits, labels = prepare_data()
    # digits, labels = load_digits('digits.png')

    print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    my_digit = digits[0]

    
    print('Deskew images ... ')
    digits_deskewed = list(map(deskew, digits))
    
    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog()


    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    for img in digits_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)
    
    # my_hog = hog.compute(my_digit)

    print('Spliting data into training (90%) and test set (10%)... ')
    # train_n=int(4999)
    train_n=int(0.9*len(hog_descriptors))
    digits_train, digits_test = np.split(digits_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])
    
    
    print('Training SVM model ...')
    model = svmInit()
    svmTrain(model, hog_descriptors_train, labels_train)

    # # my_hog = np.squeeze(my_hog)
    # my_hog = np.swapaxes(my_hog, 1, 0)
    # four = preprocess(hog)
 
    # predictions = model.predict(four)
    # print predictions
    # exit(0)
    print('Evaluating model ... ')
    vis = svmEvaluate(model, digits_test, hog_descriptors_test, labels_test)

    cv2.imwrite("digits-classification.jpg",vis)
    cv2.imshow("Vis", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
