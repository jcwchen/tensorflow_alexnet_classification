import cv2, random
import numpy as np
import IPython
import multiprocessing as mp

# Data format
class ImageClassData:
    def __init__(self, image, label):
        # CIFAR-10 raw image to rgb images and resize to 227*227
        self.image = transform_img(rawtorgb(image))
        self.label = label

# Dataset for random sampling
class Dataset:
    def __init__(self, data):
        self.data = []
        for d in data:
            self.data.append(d)
        self.data_length = len(self.data)
        self.reset_sample()
    def __len__(self):
        return self.data_length

    def sample(self, batch_size):
        data_list = []
        isnextepoch = False 
        if self.sample_top + batch_size < self.data_length:
            self.sample_top += batch_size
            data_list = self.data[self.sample_top-batch_size:self.sample_top]
        else:
            isnextepoch = True
            data_list = self.data[self.sample_top:self.data_length]
            self.sample_top = self.data_length
        x = []
        y = []
        for d in data_list: 
            x.append(d.image)
            y.append(d.label)
        return np.array(x), np.array(y), isnextepoch

    def reset_sample(self):
        self.sample_top = 0
        self.data_length = len(self.data)
        random.shuffle(self.data)

def rawtorgb(rawimg, width = 32, height = 32):
    """ Transfer 1d raw image to rgb image"""
    img = []
    for rgb in xrange(3):
        single = []
        for h_i in xrange(height):
            single.append(rawimg[rgb*width*height+h_i*width:rgb*width*height+(h_i+1)*(width)])
        img.append(single)
    img = np.array(img)
    return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)

def transform_img(img, img_width = 227 , img_height= 227):
    """ crop to 227*227 for alexnet """
    #Histogram Equalization
    """
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    """

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img