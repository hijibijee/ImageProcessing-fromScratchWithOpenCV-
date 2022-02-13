# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:46:00 2022

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
    plt.imshow(img, cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def saveImage(dir_, img):
    cv2.imwrite(dir_, img)

def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getLaplacian3X3():
    # Here at the center positive value is used, so we need to the filtered result with input to get sharpened image
    return np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    #return np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

def convolve(img, kernel):
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    n = img.shape[0]
    m = img.shape[1]
    output = np.zeros((n,m), dtype = int)
    for x in range(n):
        for y in range(m):
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    if x - s < n and x - s >= 0 and y - t < m and y - t >= 0:
                        output[x][y] += kernel[s + a][t + b] * img[x - s][y - t] # filter function for averaging
                    else:
                        output[x][y] += kernel[s + a][t + b] * 0 # we didn't pad the input image with zero explicitly
            
    return output

def scaleImage(img, K = 255):
    # when we add or subtract two image, pixels values can get less than 0 or greater than 255
    # We can use clipping (set negative values to 0 and > 255 values to 255)
    # Or we can use scaling as follows
    # g_m = g - g.min()
    # g_s = K * (g / g.max())
    
    g_m = img - img.min()
    g_s = K * (g_m / g_m.max())
    
    return g_s.astype(int)

def clipImage(img):
    clippedImage = np.zeros(img.shape, dtype = int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            clippedImage[i][j] = max(0, img[i][j])
            clippedImage[i][j] = min(255, clippedImage[i][j])
    
    return clippedImage

def main():
    input_ = toGray(readImage("input.jpg"))
    display(input_, "Input image")
    
    kernel = getLaplacian3X3()
    filteredImage = convolve(input_,kernel) #filtering with laplacian kernel
     
    display(clipImage(filteredImage), "Filtered Image (clipped)")
    display(scaleImage(filteredImage), "Filtered Image (scaled)")
    
    sharpedImage = input_ + filteredImage # if kernel center is negative -> subtract filter image
    
    display(clipImage(sharpedImage), "Sharped Image (clipped)")
    saveImage("output_clipped.jpg", clipImage(sharpedImage))
    
    display(scaleImage(sharpedImage), "Sharped Image (scaled)")
    saveImage("output_scaled.jpg", scaleImage(sharpedImage))
    
    wait()
    
main()
    
    