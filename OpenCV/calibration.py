import numpy as np
import cv2 as cv

tagDict = cv.aruco.Dictionary_get( cv.aruco.DICT_4X4_50 )
detectParams = cv.aruco.DetectorParameters_create()

green = (0, 255, 0)
red = (0, 0, 255)
width = 2

# Set up webcam and related parameters
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




	cv.imshow('Video', frame)

	inp = cv.waitKey(1)
	if inp == ord('q'):
		break
	if inp == ord('f'):
		webcam.set(cv.CAP_PROP_AUTOFOCUS, (webcam.get(cv.CAP_PROP_AUTOFOCUS)+1)%2 )


webcam.release()
cv.destroyAllWindows()
