# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar

x_markers = 4
y_markers = 4
marker_width = 3
marker_separation = 2
first_marker = 6

dictionary = ar.Dictionary_get( cv.aruco.DICT_4X4_50 )
board = ar.GridBoard_create(x_markers, y_markers, marker_width, marker_separation, dictionary, first_marker)

margin = 0
img = board.draw((360*2,360*2), margin)

#Dump the calibration board to a file
cv.imwrite('grid_board_2.png',img)
