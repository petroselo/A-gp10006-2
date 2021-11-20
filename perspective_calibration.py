# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar

# User-defined constants
import constants as C

# Returns a table width, height, and perspective matrix.
# N frames to average perspective over.
def get_table_camera_transform(board, webcam, detect_params, avg_frames, cam_mtx, dist_coeffs):

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

	unit_dst = np.float32([[0,1], [0,0], [1, 0], [1,1]])
	bounds = np.array([9, 9, 0])
	captured_frames = 0
	capturing = False
	avg_table_corners = np.zeros((4,2), dtype='float32')
	while captured_frames < avg_frames:
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


		cv.imshow(C.TITLE_PC, frame)

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
			if inp == ord('c'):
				capturing = True
	small_side = min(bounds[0], bounds[1])
	dimensions = (600 * np.array([bounds[0]/small_side, bounds[1]/small_side])).astype('int')

	CM = cv.getPerspectiveTransform((avg_table_corners / avg_frames).astype('float32'), (unit_dst * dimensions).astype('float32'))
	cv.destroyWindow(C.TITLE_PC)
	#CM is float64
	return CM, dimensions
