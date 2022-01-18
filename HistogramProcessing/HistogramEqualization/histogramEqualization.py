# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 05:55:22 2022

@author: Taif
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def readImage(dirOfFile):
    img = cv2.imread(dirOfFile)
    return img

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

def histogramEqualizationOf(img):
    n = img.shape[1]
    m = img.shape[2]
    output = img.copy()

    for c in range(img.shape[0]):
        freq = np.zeros(256, int)
        for i in range(n):               # caluculating frequencies of each intensity
            for j in range(m):
                freq[img[c].item(i, j)] += 1
        
        pdf = np.zeros(256, float)
        
        for i in range(256):
            pdf[i] = freq[i] / (n * m)
        
        cdf = np.zeros(256, float)
        cdf[0] = pdf[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + pdf[i] # calculating sum of p_k for the formula
        
        for i in range(n):
            for j in range(m):
                output[c].itemset((i, j), 255 * cdf[img[c].item(i, j)])
    
    return output

def main():
    input_ = readImage("input.jpg")
    display(input_, "Input image")
    showHistogramOf(input_, "Histogram of input image")
    
    output_ = histogramEqualizationOf(input_)
    display(output_, "Output image")
    showHistogramOf(output_, "Histogram of output image")
    
    saveImage("output.jpg", output_)
    
    wait()
    
main()