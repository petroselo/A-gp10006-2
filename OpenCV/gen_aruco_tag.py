import numpy as np
import cv2 as cv

dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
border = 1
fid_id = 0

fiducial = np.zeros((400,400,1), dtype='uint8')

cv.aruco.drawMarker(dictionary, fid_id, 400, fiducial, border )

cv.imshow('Fiducial', fiducial)
cv.waitKey(0)
