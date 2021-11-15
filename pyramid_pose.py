#!/usr/bin/env python3

# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar

# User modules
from calibrate import calibrate

# User-defined constants
import constants as C

VIDEO_TITLE = 'Preview'

pyramid_points = 18 * np.array([
[0,0,0],
[0,1,0],
[1,1,0],
[1,0,0],
[.5, .25, 0],
[.25, .5, 0],
[.5, .75, 0],
[.75, .5, 0],
[.5, .5, .25]
])

unit_corner_points = np.array([
[0.,0,0],
[0.,1,0],
[1.,1,0],
[1.,0,0]
])

dst = np.float32([[0.,200], [0.,0], [200, 0.0], [200.,200]])

def main():
	detect_params = cv.aruco.DetectorParameters_create()

## Set up webcam
	webcam = cv.VideoCapture(0)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, 800)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
	#webcam.set(cv.CAP_PROP_FPS, 25)
	webcam.set(cv.CAP_PROP_AUTOFOCUS, 0) 	#Disable autofocus
	if not webcam.isOpened():
		print('Failed to open camera.')
		exit()

## Initial calibration
	cam_mtx = C.INITIAL_CALIBRATION_CM
	dist_coeffs = C.INITIAL_CALIBRATION_DC

## Main loop
	while True:

		ret, frame = webcam.read()
		if not ret:
			print('Video finished')
			break

		(corners, ids, rejected) = cv.aruco.detectMarkers(frame, C.DICTIONARY, parameters = detect_params)

		if ids is not None and len(ids) > 0:
			valid, rvec, tvec = ar.estimatePoseBoard(corners, ids, C.BOARD6, cam_mtx, dist_coeffs, None, None, False)
			if valid:


							#extract board into separate stream
				corns, _ = cv.projectPoints(18*unit_corner_points, rvec, tvec, cam_mtx, dist_coeffs)
				newcorns = corns.reshape(-1,2).astype('float32')

				PM = cv.getPerspectiveTransform(newcorns, dst)
				board6 = cv.warpPerspective(frame, PM, (200, 200))
				cv.imshow('Extract', board6)

				# Draw custom pyramid
				pts, _ = cv.projectPoints(pyramid_points, rvec, tvec, cam_mtx, dist_coeffs)
				newpts = np.int32(pts).reshape(-1, 2)
				cv.drawContours(frame, [newpts[0:4]], -1, (0,200,200), -1)
				for p in range(4):
					cv.line(frame, tuple(newpts[4+p]), tuple(newpts[8]), (100, 100, 100), 2)

	# draw default axis markers
				ar.drawAxis(frame, cam_mtx, dist_coeffs, rvec, tvec, 6)




		cv.imshow(VIDEO_TITLE, frame)


		inp = cv.waitKey(1)
		if inp == ord('q'):
			print(frame.shape)
			break
		if inp == ord('f'):
			print(webcam.get(cv.CAP_PROP_FOCUS))
			print(webcam.get(cv.CAP_PROP_ZOOM))
			webcam.set(cv.CAP_PROP_AUTOFOCUS, (webcam.get(cv.CAP_PROP_AUTOFOCUS)+1)%2 )

		if inp == ord('c'):
			ret, (ret_cam_mtx, ret_dist_coeffs) = calibrate(webcam, C.CALIBRATION_NUMBER)
			if ret:
				cam_mtx, dist_coeffs = ret_cam_mtx, ret_dist_coeffs
		if inp == ord('r'):
			cam_mtx = np.loadtxt(C.CM_FILENAME)
			dist_coeffs = np.loadtxt(C.DC_FILENAME)
		if inp == ord('w'):
			np.savetxt(C.CM_FILENAME, cam_mtx)
			np.savetxt(C.DC_FILENAME, dist_coeffs)

	webcam.release()
	cv.destroyAllWindows()


if __name__ == "__main__":
	main()
