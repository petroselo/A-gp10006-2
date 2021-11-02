import numpy as np
import cv2 as cv

tagDict = cv.aruco.Dictionary_get( cv.aruco.DICT_4X4_50 )
detectParams = cv.aruco.DetectorParameters_create()

green = (0, 255, 0)
red = (0, 0, 255)
width = 2

webcam = cv.VideoCapture(0)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, 800) #800
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 600) #600
#webcam.set(cv.CAP_PROP_FPS, 25)

#Disable autofocus
webcam.set(cv.CAP_PROP_AUTOFOCUS, 0)
print(webcam.get(cv.CAP_PROP_FOCUS))


if not webcam.isOpened():
	print('Failed to open camera.')
	exit()

while True:

	ret, frame = webcam.read()
	if not ret:
		print('Video finished')
		break

	(allCorners, ids, rejected) = cv.aruco.detectMarkers(frame, tagDict, parameters = detectParams)

	if len(allCorners) > 0:
		fids = ids.flatten()
		for (tagCorners, fid) in zip(allCorners, fids):
			# Convert to ints in the frame.
			(tl, tr, br, bl) = pts = tagCorners.reshape((4, 2)).astype(np.int32)

			#Draw lines to show markers and rejected options.
			cv.polylines(frame, [pts], True, green, width)

#	for reject in rejected:
#		pts = reject.reshape((4,2)).astype(np.int32)
#		cv.polylines(frame, [pts], True, red, width)
#		cv.line(frame, pts[0], pts[2],red, width)
#		cv.line(frame, pts[1], pts[3],red, width)


	cv.imshow('Video', frame)

	inp = cv.waitKey(1)
	if inp == ord('q'):
		print(frame.shape)
		break
	if inp == ord('f'):
		print(webcam.get(cv.CAP_PROP_FOCUS))
		print(webcam.get(cv.CAP_PROP_ZOOM))
		webcam.set(cv.CAP_PROP_AUTOFOCUS, (webcam.get(cv.CAP_PROP_AUTOFOCUS)+1)%2 )


webcam.release()
cv.destroyAllWindows()
