#!/usr/bin/env python3

# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar, imshow
from numpy.core.fromnumeric import shape

# User-defined constants
import constants as C

# User modules
#from camera_calibration import calibrate
from perspective_calibration import get_table_camera_transform
from projector_calibration_manual import calibrate_projector
from Logic_cards import Logic_card

## Initial calibration values
cam_mtx = C.INITIAL_CALIBRATION_CM
dist_coeffs = C.INITIAL_CALIBRATION_DC

def main():

	webcam, detect_params = initial_setup()

	CM = np.eye(3, dtype='float64')
	PM = np.eye(3, dtype='float64')
	dimensions = np.array([800, 600])

	projector_blank = False

	logic_cards = []

	proj_window = cv.namedWindow(C.PROJ_WINDOW, cv.WND_PROP_FULLSCREEN)
	proj_img = np.zeros(shape=(C.PROJ_HEIGHT, C.PROJ_WIDTH, 3), dtype=np.uint8) + 255
	cv.putText(proj_img, 'Hello World', (960,540), C.FONT, 1, C.BLUE, 2)
	cv.imshow(C.PROJ_WINDOW, proj_img)

	table_overlay = np.zeros(shape=(dimensions[1]*C.TABLE_OVERLAY_FACTOR, dimensions[0]*C.TABLE_OVERLAY_FACTOR, 3), dtype=np.uint8)

## Main loop
	while True:

		ret, cam_frame = webcam.read()
		if not ret:
			print('Video finished')
			break

		table_frame = cv.warpPerspective(cam_frame, CM, dimensions)

		(allCorners, ids, rejected) = cv.aruco.detectMarkers(table_frame, C.DICTIONARY, parameters = detect_params)

		#construct list of logic cards and their positions.

		#send that information off to another function to analyse the logic and work out what to do about it and draw that on the frame.

		if len(allCorners) > 0:
			fids = ids.flatten()
			markers = zip(fids, allCorners)
			for (fid, raw_corners) in markers:
				# If in logic card range of markers.
				if fid > 19 and fid < 28:
					corners = raw_corners.reshape((4, 2)).astype(np.int32)
					logic_cards.append( Logic_card(fid, corners))

		# Loop through logic cards connecting each input to closest output within snapping distance.
		for lc in logic_cards:
			for inp in lc.inps:
				for lcc in logic_cards:
					if lc is not lcc: # exclude current card
						for outp in lcc.outps:
							dist = np.linalg.norm(outp.pos - inp.pos)
							if dist < lc.snap_distance and ( (inp.conn is None) or (dist < np.linalg.norm(inp.pos - inp.conn.pos) )):
								inp.conn = outp

		# Evaluate the state of each card.
		for lc in logic_cards:
			if not lc.evaluated:
				lc.evaluate()

		# Loop through logic cards drawing the state
		for lc in logic_cards:

			for inp in lc.inps:
				cv.line(table_frame, tuple(inp.pos.astype('int')), tuple((inp.pos+lc.xvec).astype('int')), C.BLACK)
				cv.circle(table_frame, tuple(inp.pos.astype('int')), int(0.25*lc.scale), C.BLACK)
				if inp.conn is not None:
					cv.line(table_frame, tuple(inp.pos.astype('int')), tuple((inp.conn.pos).astype('int')), C.BLACK)

			for o in lc.outps:
				cv.circle(table_frame, tuple(o.pos.astype('int')), int(0.25*lc.scale), C.BLUE if o.val>1 else (C.GREEN if o.val==1 else C.RED), -1 if o.val<2 else 1)
				cv.line(table_frame, tuple(o.pos.astype('int')), tuple((o.pos-lc.xvec).astype('int')), C.BLUE, 2 if o.val<2 else 1)

		logic_cards.clear()

		cv.imshow(C.VIDEO_TITLE, table_frame)


		# Quitting condition
		inp = cv.waitKey(1)
		if inp == ord('q'):
			break

		# Generate, Save and Load previous perspective calibration
		if inp == ord('c'):
			CM, dimensions = get_table_camera_transform(C.BOARD6_2, webcam, detect_params, avg_frames=20,
																									cam_mtx=cam_mtx, dist_coeffs=dist_coeffs)
			table_overlay = np.zeros(shape=(dimensions[0]*C.TABLE_OVERLAY_FACTOR, dimensions[1]*C.TABLE_OVERLAY_FACTOR, 3), dtype=np.uint8)
		
		if inp == ord('l'):
			CM = np.loadtxt('PerspectiveMatrix.txt')
			dimensions = np.loadtxt('Dimensions.txt', dtype='int64')
			table_overlay = np.zeros(shape=(dimensions[1]*C.TABLE_OVERLAY_FACTOR, dimensions[0]*C.TABLE_OVERLAY_FACTOR, 3), dtype=np.uint8)
		if inp == ord('s'):
			np.savetxt('PerspectiveMatrix.txt', CM)
			np.savetxt('Dimensions.txt', dimensions, fmt='%u')
		
		#Toggle projector window to fullscreen.
		if inp ==ord('f'):
			if cv.getWindowProperty(C.PROJ_WINDOW, cv.WND_PROP_FULLSCREEN) == cv.WINDOW_NORMAL:
				cv.setWindowProperty(C.PROJ_WINDOW,cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
			else:
				cv.setWindowProperty(C.PROJ_WINDOW,cv.WND_PROP_FULLSCREEN,cv.WINDOW_NORMAL)

		# Projector calibration
		if inp == ord('p'):
			PM = calibrate_projector(webcam, CM, dimensions, avg_frames=20, detect_params=detect_params)
			print(PM)
			
		if inp == ord('b'):
			if projector_blank:
				imshow(C.PROJ_WINDOW, np.zeros(shape=C.PROJ_SHAPE))
			else:
				imshow(C.PROJ_WINDOW, np.zeros(shape=C.PROJ_SHAPE))
			projector_blank = not projector_blank

	webcam.release()
	cv.destroyAllWindows()


def initial_setup():
	detect_params = cv.aruco.DetectorParameters_create()

	# Set up webcam
	webcam = cv.VideoCapture(2)
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
