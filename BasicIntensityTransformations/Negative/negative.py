# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 02:42:48 2022

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
    
def saveImage(dir, img):
    cv2.imwrite(dir, img)

def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def negativeOf(img):
    n = img.shape[0]
    m = img.shape[1]
    output = np.zeros(img.shape)

    for i in range(n):
        for j in range(m):
            output[i][j] = 255 - img[i][j]
    
    return output

def main():
    input = toGray(readImage("input.jpg"))
    display(input, "Input Image")
    output = negativeOf(input)
    display(output, "Output Image")
    saveImage("output.jpg", output)
    wait()
    
main()
    
    