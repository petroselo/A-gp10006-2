# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar
from numpy.core.fromnumeric import shape

# User-defined constants
import constants as C

def calibrate_projector():

	# Draw a white projector image.
	calib_img = np.zeros(shape=(C.PROJ_HEIGHT, C.PROJ_WIDTH, 3), dtype=np.uint8) + 255

	board = C.PROJECTOR_BOARD

	# Draw board on 880 x 880 image. This gives 100px margin & allows grid of length 11 (4*2 markers +3 gaps) to fit pixel perfectly.
	extent = 880
	margin = 0
	board_img = board.draw((extent,extent), margin)

	# Edges
	left = C.PROJ_WIDTH//2 - extent//2
	right = left + extent
	top = C.PROJ_HEIGHT//2 - extent//2
	bot = top + extent

	print(left, right, top, bot)

	# Insert Board image into projector output.
	calib_img[ top:bot, left:right, : ] = board_img.reshape((extent, extent, 1))

	cv.imshow(C.PROJ_WINDOW, calib_img)