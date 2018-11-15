import dlib
import struct
import cv2 as cv
import numpy as np

class Detector(object):
    def __init__(self, options = None, loadPath = None):
        if loadPath is not None: #Loading exisiting hog model
            self.detector = dlib.simple_object_detector(loadPath)
        
        if options is None:
            self.options = dlib.simple_object_detector_training_options()
        else:
            self.options = options


    def read_database(self, filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


    def prepare_data(self):
        
        imPaths = []
        images_rgb = []

        images = self.read_database('database/emnist-letters-test-images-idx3-ubyte')
        labels = self.read_database('database/emnist-letters-test-labels-idx1-ubyte')
        rotate = cv.getRotationMatrix2D((14,14),-90,1)
        for i in range(len(images)):
            images[i] = cv.warpAffine(images[i],rotate,(28,28))
            images[i] = cv.flip(images[i], 1)
            # cv.imshow('sd', images[i])
            # cv.waitKey(0)
            images_rgb.append(cv.cvtColor(images[i], cv.COLOR_GRAY2RGB))
            cv.imwrite('training_data/img' + str(i) + '.jpeg', images[i])
            imPaths.append('training_data/img' + str(i) + '.jpeg')  

        annotations = [0, 0, 28, 28] * i
        annotations = np.array(annotations)
        imPaths = np.array(imPaths, dtype = 'unicode')
        # np.save('annotations', annotations)
        # np.save('imPaths', imPaths)
        
        self.images = images_rgb
        self.annotations = annotations

    def fit(self, vis, savePath = None):
        print 'before'
        self.detector = dlib.train_simple_object_detector(self.images, self.annotations, self.options)
        print 'after'

        if vis:
            win = dlib.image_window()
            win.set_image(self.detector)
            dlib.hit_enter_to_continue()
            print 'vis'
        
        if savePath is not None:
            print 'before'
            self.detector.save(savePath)
            print 'after'
        
        return self



if __name__ == "__main__":
    det = Detector()
    det.prepare_data()
    # print type(det.annotations[0])
    det.fit(vis = True)

