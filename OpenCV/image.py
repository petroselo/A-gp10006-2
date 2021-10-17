##!pip3 install opencv-python

#imports
import numpy as np
import cv2 as cv

#Load in 2 images at half scale each dimension
landscape_img = cv.imread('landscape.jpg', cv.IMREAD_REDUCED_COLOR_2 ) #cv.IMREAD_UNCHANGED
extract_img = cv.imread('extract.png', cv.IMREAD_REDUCED_COLOR_2)
if landscape_img is None:
    print('Failed to load landscape')
if extract_img is None:
    print('Failed to load extract')

match = (cv.matchTemplate(landscape_img, extract_img, cv.TM_CCOEFF_NORMED)

, _, minLoc, maxLoc = cv.minMaxLoc(match)
print(minLoc, maxLoc)

#print(landscape_img.dtype)
#print(match.dtype)

#cv.imshow('Title', match)
#cv.waitKey(0)

print(cv.imwrite('result.png', match))
print('done')
