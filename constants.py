import numpy as np
import cv2 as cv
from cv2 import aruco as ar

DICTIONARY = ar.Dictionary_get(ar.DICT_4X4_50)

# Default font to use
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_WIDTH = 1

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
LINE_WIDTH = 2



# Initial calibration
INITIAL_CALIBRATION_CM = np.array(
																		[[600., 0, 300],
																		 [0, 800., 400],
																		 [0, 0, 1.]])

INITIAL_CALIBRATION_DC = np.array([0.0, 0, 0, 0, 0])
CALIBRATION_NUMBER = 15
CM_FILENAME = 'calib_camera_matrix.txt'
DC_FILENAME = 'calib_dist_coeffs.txt'

#should be in the right units. need to verify this
x_markers = 2
y_markers = 2
marker_width = 3
marker_separation = 12
first_marker = 6

dictionary = ar.Dictionary_get( cv.aruco.DICT_4X4_50 )
BOARD6 = ar.GridBoard_create(x_markers, y_markers, marker_width, marker_separation, dictionary, first_marker)
