# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 04:00:43 2022

@author: Taif
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def readImage(dirOfFile):
    img = cv2.imread(dirOfFile)
    return img

def toGray(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

def display(img, title = 'Image', cmap_type = 'gray'):
    plt.imshow(img, cmap = cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def showHistogramOf(img, title = "histogram"):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title(title)
    plt.show()
    
def saveImage(dir_, img):
    cv2.imwrite(dir_, img)

def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contrastStretch(img, s1, s2):
    n = img.shape[1]
    m = img.shape[2]
    output = img.copy()
    
    for c in range(img.shape[0]):
        r2 = img[c].max()
        r1 = img[c].min()

        for i in range(n):
            for j in range(m):
                output[c].itemset((i, j), (s2 - s1) * (img[c].item(i, j) - r1) / (r2 - r1) + s1)
    
    return output

def main():
    input_ = readImage("input.jpg")
    display(input_, "Input image")
    showHistogramOf(input_, "Histogram of input image")
    output_ = contrastStretch(input_, 0, 255)
    display(output_, "Output image")
    showHistogramOf(output_, "Histogram of output image")
    saveImage("output.jpg", output_)
    
    wait()
    
main()