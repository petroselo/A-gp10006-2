##!pip3 install opencv-python

#imports
import numpy as np
import cv2 as cv

font = cv.FONT_HERSHEY_SIMPLEX

#Load in 2 images at half scale each dimension
landscape_img = cv.imread('landscape.jpg', cv.IMREAD_REDUCED_COLOR_2 ) #cv.IMREAD_UNCHANGED
extract_img = cv.imread('extract.png', cv.IMREAD_REDUCED_COLOR_2)
if landscape_img is None:
    print('Failed to load landscape')
if extract_img is None:
    print('Failed to load extract')

match = (cv.matchTemplate(landscape_img, extract_img, cv.TM_CCOEFF_NORMED))

_, _, minLoc, maxLoc = cv.minMaxLoc(match)
print(minLoc, maxLoc)

extract_size = np.array([extract_img.shape[1], extract_img.shape[0]])
print(extract_size)
print(minLoc + extract_size)

# Draw min
cv.rectangle(landscape_img, minLoc, minLoc + extract_size, (0,0,255), 1)
cv.putText(landscape_img, 'Min', minLoc, font, 1, (0,0,255), 1, lineType=cv.LINE_AA)
#Draw max
cv.rectangle(landscape_img, maxLoc, maxLoc + extract_size, (180,255,0), 1)
cv.putText(landscape_img, 'Max', maxLoc, font, 1, (180,255,0), 1, lineType=cv.LINE_AA)


cv.imshow('Title', landscape_img)
cv.waitKey(0)

print(cv.imwrite('result.png', match))
print('done')
