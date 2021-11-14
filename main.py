#!/usr/bin/env python3

# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar

# User modules
#from camera_calibration import calibrate
from perspective_calibration import perspective_calibrate


# User-defined constants
import constants as C

##  GLOBAL STUFF
VIDEO_TITLE = 'Table View'
TITLE_PC = 'Perspective Calibration'

unit_corner_points = np.array([
[-1,-1,0],
[-1,1,0],
[1,1,0],
[1.,-1,0]
], dtype='float32')
centre_points = np.array([
[9,9,0],
[9,9,0],
[9,9,0],
[9,9,0]
], dtype='float32')

## Initial calibration values
cam_mtx = C.INITIAL_CALIBRATION_CM
dist_coeffs = C.INITIAL_CALIBRATION_DC

def main():

	webcam, detect_params = initial_setup()

	PM = np.eye(3, dtype='float64')
	dimensions = np.array([800, 600])

## Main loop
	while True:

		ret, cam_frame = webcam.read()
		if not ret:
			print('Video finished')
			break

		table_frame = cv.warpPerspective(cam_frame, PM, dimensions)

		# do all work with table frame.

		# draw table frame, untransform table frame back to cam frame and draw that too.

		(corners, ids, rejected) = cv.aruco.detectMarkers(table_frame, C.DICTIONARY, parameters = detect_params)

#		if len(corners) > 0:
#			fids = ids.flatten()
#			markers = zip(fids, corners)
#			for (fid, corner) in markers:
#				if fid > 19 and fid < 28:
#
#				# Convert to ints in the frame.
#				#(tl, tr, br, bl) = pts = tagCorners.reshape((4, 2)).astype(np.int32)
#
#				#Draw lines to show markers and rejected options.
#				cv.polylines(table_frame, [pts], True, C.GREEN, C.LINE_WIDTH)
#

		cv.imshow(VIDEO_TITLE, table_frame)


		inp = cv.waitKey(1)
		if inp == ord('q'):
			break

		if inp == ord('p'):
			PM, dimensions = set_table_camera_transform(C.BOARD6_2, webcam, detect_params, n_frames=10)


	webcam.release()
	cv.destroyAllWindows()

# Returns a table width, height, and perspective matrix.
def set_table_camera_transform(board, webcam, detect_params, n_frames):
	unit_dst = np.float32([[0,1], [0,0], [1, 0], [1,1]])
	bounds = np.array([9, 9, 0])
	captured_frames = 0
	capturing = False
	avg_table_corners = np.zeros((4,2), dtype='float32')
	while captured_frames < n_frames:
		ret, frame = webcam.read()
		if not ret:
			print('Video finished')
			break

		corners, ids, _ = cv.aruco.detectMarkers(frame, board.dictionary, parameters = detect_params)

		# Get the pose of the board.
		if ids is not None and len(ids) > 0:
			valid, rvec, tvec = ar.estimatePoseBoard(corners, ids, board, cam_mtx, dist_coeffs, None, None, False)
			if valid:
				# Determint eh location of the bound corners in board space
				corners = centre_points + unit_corner_points*bounds
				# Find the location of the bound corners in image space
				table_corners, _ = cv.projectPoints(corners, rvec, tvec, cam_mtx, dist_coeffs)
				table_corners = table_corners.reshape(4,2).astype('float32')
				table_corners_int = table_corners.astype('int')

				if capturing:
					avg_table_corners += table_corners
					captured_frames += 1

				# Draw line around table area:
				for i in range(4):
					cv.line(frame, tuple(table_corners_int[i,:]), tuple(table_corners_int[(i+1)%4,:]), C.RED if capturing else C.BLUE, 2)


		cv.imshow(TITLE_PC, frame)

		inp = cv.waitKey(1)
		if inp == ord('q'):
			break

		if not capturing:
			# Adjust bounds
			if inp == ord('d'):
				bounds[0] += 1
			if inp == ord('a'):
				bounds[0] -= 1
			if inp == ord('w'):
				bounds[1] += 1
			if inp == ord('s'):
				bounds[1] -= 1

	# Calculate perspective transform.
			if inp == ord('p'):
				capturing = True
	small_side = min(bounds[0], bounds[1])
	dimensions = (600 * np.array([bounds[0]/small_side, bounds[1]/small_side])).astype('int')

	PM = cv.getPerspectiveTransform((avg_table_corners / n_frames).astype('float32'), (unit_dst * dimensions).astype('float32'))
	cv.destroyWindow(TITLE_PC)
	#PM is float64
	return PM, dimensions


def initial_setup():
	detect_params = cv.aruco.DetectorParameters_create()

	# Set up webcam
	webcam = cv.VideoCapture(0)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, 800)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
	#webcam.set(cv.CAP_PROP_FPS, 25)
	webcam.set(cv.CAP_PROP_AUTOFOCUS, 0) 	#Disable autofocus
	if not webcam.isOpened():
		print('Failed to open camera.')
		exit()
	return webcam, detect_params

if __name__ == "__main__":
	main()
