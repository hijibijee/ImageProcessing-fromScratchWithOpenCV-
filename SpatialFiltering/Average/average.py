# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 07:34:11 2022

@author: Taif
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def readImage(dirOfFile):
    img = cv2.imread(dirOfFile)
    return img

def toGray(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

def display(img, title = 'Image', cmap_type = 'gray'):
    plt.imshow(img, cmap_type)
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
    
def generateBoxKernelOfSize(size):
    kernel = np.ones((size, size), float)
    return kernel

def generateGaussianKernel(size, sigma = 1, K = 1):
    kernel = np.zeros((size, size))
    a = size // 2
    for s in range(-a, a + 1):
        for t in range(-a, a + 1):
            kernel[s + a][t + a] = K * math.exp(-(s*s + t*t)/(2 * sigma * sigma))
    return kernel

def correlate(img, kernel):
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    sumOfKernel = kernel.sum()
    n = img.shape[0]
    m = img.shape[1]
    output = img.copy()
    for x in range(n):
        for y in range(m):
            val = 0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    if x + s < n and x + s >= 0 and y + t < m and y + t >= 0:
                        val += kernel[s + a][t + b] * img.item(x + s, y + t) # filter function for averaging
                    else:
                        val += kernel[s + a][t + b] * 0 # we are not padding the input image with zero hence this is required
            val /= sumOfKernel
            output.itemset((x, y), val)
    return output

def convolve(img, kernel):
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    sumOfKernel = kernel.sum()
    n = img.shape[0]
    m = img.shape[1]
    output = img.copy()
    for x in range(n):
        for y in range(m):
            val = 0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    #print(s + a, t + b, x - s, y - t, x, y)
                    if x - s < n and x - s >= 0 and y - t < m and y - t >= 0:
                        val += kernel[s + a][t + b] * img.item(x - s, y - t) # filter function for averaging
                    else:
                        val += kernel[s + a][t + b] * 0 # we are not padding the input image with zero hence this is required
            val /= sumOfKernel
            output.itemset((x, y), val)
    return output
                                               
def main():
    input_ = toGray(readImage("input.jpg"))
    display(input_, "Input image")
    
    kernel = generateBoxKernelOfSize(3)

    output_ = correlate(input_, kernel)
    display(output_, "Output image with kernel size 3")
    saveImage("outputWithKernel_3x3.jpg", output_)
    
    kernel = generateBoxKernelOfSize(5)
    output_ = correlate(input_, kernel)
    display(output_, "Output image with kernel size 5")
    saveImage("outputWithKernel_5x5.jpg", output_)
    
    kernel = generateBoxKernelOfSize(11)
    output_ = correlate(input_, kernel)
    display(output_, "Output image with kernel size 11")
    saveImage("outputWithKernel_11x11.jpg", output_)
    
    kernel = generateGaussianKernel(3)
    output_ = correlate(input_, kernel)
    display(output_, "Output image with gaussian kernel size 3")
    saveImage("outputWithGaussianKernel_3x3.jpg", output_)
    
    wait()
    
main()
    
    