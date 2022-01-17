# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:05:24 2022

@author: Taif
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def normalize(x, min_, max_, tarMin, tarMax):
    assert max_ - min_ != 0  
    fx = ((tarMax - tarMin) * (x - min_) / (max_ - min_)) + tarMin
    return fx

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
    
def saveImage(dir_, img):
    cv2.imwrite(dir_, img)

def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gammaOf(img, gamma = 0.4, c = 1):
    n = img.shape[0]
    m = img.shape[1]
    output = np.zeros(img.shape)

    for i in range(n):
        for j in range(m):
            output[i][j] = c * math.pow(img[i][j], gamma)
            output[i][j] = normalize(output[i][j], 0, c * math.pow(255, gamma), 0, 255)
    
    return output

def main():
    input_image1 = toGray(readImage("input1.jpg"))
    input_image2 = toGray(readImage("input2.jpg"))
    display(input_image1, "input 1")
    display(input_image2, "input 2")
    
    output1 = gammaOf(input_image1, 0.3) 
    display(output1, "Output Image 1")
    saveImage("output1.jpg", output1)
    
    output2 = gammaOf(input_image2, 3.0) 
    display(output2, "Output Image 2")
    saveImage("output2.jpg", output2)
    
    wait()
    
main()