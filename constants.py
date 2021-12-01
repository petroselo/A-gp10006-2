import numpy as np
import cv2 as cv
from cv2 import aruco as ar

# Logic gate enum ersatz. obv 0 and 1 are themselves
FLOATING = 2
UNKNOWABLE = 3

##  GLOBAL STUFF
VIDEO_TITLE = 'Table View'
TITLE_PC = 'Perspective Calibration'
PROJ_WINDOW = 'Projector output'

DICTIONARY = ar.Dictionary_get(ar.DICT_4X4_50)

# Default font to use
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_WIDTH = 1

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)

LINE_WIDTH = 2

# Camera calibration
INITIAL_CALIBRATION_CM = np.array([[600.,   0., 300.],
								   [  0., 800., 400.],
								   [  0.,   0.,   1.]])

INITIAL_CALIBRATION_DC = np.array([0.0, 0, 0, 0, 0])
CALIBRATION_NUMBER = 15
CM_FILENAME = 'calib_camera_matrix.txt'
DC_FILENAME = 'calib_dist_coeffs.txt'

# Camera to table transform

#should be in the right units. need to verify this
x_markers = 2
y_markers = 2
marker_width = 3
marker_separation = 12
first_marker = 6

dictionary = ar.Dictionary_get( cv.aruco.DICT_4X4_50 )
BOARD6 = ar.GridBoard_create(x_markers, y_markers, marker_width, marker_separation, dictionary, first_marker)
BOARD6_2 = ar.GridBoard_create(4, 4, 3, 2, dictionary, 6)

# 4x4 board with markers width 2 and gap between width 1. Total side length 11.
PROJ_BOARD = ar.GridBoard_create(2, 2, 3, 5, dictionary, 46)
PROJ_BOARD_SIDELENGTH = 11

# Projector 
PROJ_WIDTH = 1920
PROJ_HEIGHT = 1080
PROJ_SHAPE = (PROJ_HEIGHT, PROJ_WIDTH)
PROJ_SCREEN_DIMENSIONS = np.array([PROJ_WIDTH, PROJ_HEIGHT], dtype='int')

# Ratio of seen table image to hi-def image on which drawing takes place.
TABLE_OVERLAY_FACTOR = 2