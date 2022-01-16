# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 04:51:25 2022

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

def logOf(img, c = 1):
    n = img.shape[0]
    m = img.shape[1]
    output = np.zeros(img.shape)

    for i in range(n):
        for j in range(m):
            output[i][j] = c * math.log(1 + img[i][j])
            output[i][j] = normalize(output[i][j], 0, c * math.log(256), 0, 255)
    
    return output

def main():
    input_image = toGray(readImage("input.jpg"))
    display(input_image, "Input Image")
    output = logOf(input_image, 1) 
    display(output, "Output Image")
    saveImage("output.jpg", output)
    wait()
    
main()