import numpy as np
import cv2 as cv
from cv2 import aruco as ar

import constants as C

# Start a calibration loop.
# Webcam: the VideoCapture
# N how many frames to take
# Returns: success,
def calibrate(webcam, n):

	xSquares = 3
	ySquares = 3
	squareLength = 2
	markerLength = 1

	calibration_dictionary = ar.Dictionary_get( cv.aruco.DICT_4X4_50 )
	calibration_board = ar.CharucoBoard_create(xSquares, ySquares, squareLength, markerLength, calibration_dictionary)

	frames_captured = 0
	calibCorners = []
	calibIds = []

	capture_next_frame = False

	calibration_window_name = 'Calibration'
	margin = 10

	while frames_captured < n:
		ret, frame = webcam.read()
		if not ret:
			print('Video finished')
			return False, None

		grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		if capture_next_frame:
			(corners, ids, rejected) = ar.detectMarkers(grey, calibration_dictionary)

			if len(corners) > 0:
				(retval, chCorners, chIds) = ar.interpolateCornersCharuco(corners, ids, grey, calibration_board)
				if retval and chCorners is not None and chIds is not None and len(chCorners) > 3:
					calibCorners.append(chCorners)
					calibIds.append(chIds)
					frames_captured += 1

					ar.drawDetectedMarkers(grey, corners , ids)

		cv.putText(grey, f'{frames_captured} / {n}', (margin,grey.shape[0]-margin), C.FONT, 1, (0,0,0), 1, lineType=cv.LINE_AA)
		cv.imshow(calibration_window_name, grey)
		inp = cv.waitKey(1)

		capture_next_frame = (inp == ord('c'))

		if inp == ord('x') or inp == ord('q'):
			return False, None


	#Calibration fails for lots of reasons. Release the video if we do
	try:
		(calibRetval, cameraMatrix, distCoeffs, rvecs, tvecs) = ar.calibrateCameraCharuco(calibCorners, calibIds, calibration_board, grey.shape, None, None)
	except:
		return False, None

	if not calibRetval:
		return False, None

	cv.destroyWindow(calibration_window_name)

	print("Camera Matrix")
	print(cameraMatrix)

	print("Dist Coeffs")
	print(distCoeffs)

	return True, (cameraMatrix, distCoeffs)
