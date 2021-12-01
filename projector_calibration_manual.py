# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar
from numpy.core.fromnumeric import shape

# User-defined constants
import constants as C

def calibrate_projector(webcam, CM, table_frame_dimensions, avg_frames, detect_params):

	cam_mtx, dist_coeffs = C.INITIAL_CALIBRATION_CM, C.INITIAL_CALIBRATION_DC

	### Draw the calibration board on the projector image.
	# Draw board on 880 x 880 image. This gives 100px margin & allows grid of length 11 (4*2 markers +3 gaps) to fit pixel perfectly.
	calib_img = np.zeros(shape=(C.PROJ_HEIGHT, C.PROJ_WIDTH, 3), dtype=np.uint8) + 255
	board = C.PROJ_BOARD
	extent = 880 
	margin = 0
	board_img = board.draw((extent,extent), margin)
	# Edges
	left = C.PROJ_WIDTH//2 - extent//2
	right = left + extent
	top = C.PROJ_HEIGHT//2 - extent//2
	bot = top + extent
	# Insert Board image into projector output.
	calib_img[ top:bot, left:right, : ] = board_img.reshape((extent, extent, 1))

	print('Showing projector calibration board')
	cv.imshow(C.PROJ_WINDOW, calib_img)

	# Board corners in projector image coordinate system
	destination_corners = np.array([
	[left,  bot],
	[left,  top],
	[right, top],
	[right, bot] 
	], dtype='float32')

	board_corners = np.array([
	[0,0,0],
	[0,1,0],
	[1,1,0],
	[1.,0,0]
	], dtype='float32') * 11

	avg_proj_corners = np.zeros((4,2), dtype='float32')

	captured_frames = 0
	capturing = False

	while captured_frames < avg_frames:
		ret, frame = webcam.read()
		if not ret:
			print('Video finished')
			break

		table_frame = cv.warpPerspective(frame, CM, table_frame_dimensions)

		corners, ids, _ = cv.aruco.detectMarkers(table_frame, board.dictionary, parameters = detect_params)

		# 46 47   and order = (tl, tr, br, bl)
		# 48 49

		# Get the pose of the board in a horrible manual way. 
		# But the gridboard estimate pose is being useless.
		if ids is not None and all([i in ids for i in range(46,50)]):
			proj_corners=np.zeros(shape=(4,2), dtype=np.float32)
			for corner, id in zip(corners, ids):
				corner = corner.reshape(4,2)
				if id == 46:
					proj_corners[1] = corner[0]
				elif id == 47:
					proj_corners[2] = corner[1]
				elif id == 48:
					proj_corners[0] = corner[3]
				elif id == 49:
					proj_corners[3] = corner[2]

			proj_corners_int = proj_corners.astype('int')

			if capturing:
				avg_proj_corners += proj_corners
				captured_frames += 1

			# Draw line around something:
			for i in range(4):
				cv.line(table_frame, tuple(proj_corners_int[i,:]), tuple(proj_corners_int[(i+1)%4,:]), C.RED if capturing else C.BLUE, 2)

		# else:
		# 	print("No projector calibration board in view.")

		inp = cv.waitKey(1)
		if inp == ord('q'):
			break
		if inp == ord('p'):
			capturing = True

		#Toggle projector window to fullscreen.
		if inp ==ord('f'):
			if cv.getWindowProperty(C.PROJ_WINDOW, cv.WND_PROP_FULLSCREEN) == cv.WINDOW_NORMAL:
				cv.setWindowProperty(C.PROJ_WINDOW,cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
			else:
				cv.setWindowProperty(C.PROJ_WINDOW,cv.WND_PROP_FULLSCREEN,cv.WINDOW_NORMAL)

		cv.imshow(C.VIDEO_TITLE, table_frame)

	PM = cv.getPerspectiveTransform((avg_proj_corners / avg_frames * C.TABLE_OVERLAY_FACTOR).astype('float32'), destination_corners)
	return PM

