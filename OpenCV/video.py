import cv2 as cv
import numpy as np
import random as rd

#Video writer for saving videos
fourcc = cv.VideoWriter_fourcc(*'FMP4') #option for encoding
#fps, width, height, isColor seems to be essential as well.
random_writer = cv.VideoWriter('colour.avi', fourcc, 5, (20, 20), isColor = True)
writer = cv.VideoWriter('output.avi', fourcc, 30, (320, 576), isColor = False)

# Can be a device (simple integer) or a file name.
# capture is a VideoCapture object
capture = cv.VideoCapture('pride.mp4')
if not capture.isOpened():
	print('Failed to open camera.')
	exit()

while True:
	ret, frame = capture.read()

	if not ret:
		print('Video finished')
		break

	#perform something on the frame
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	invert = cv.flip(-(2*gray-128)+128,0)
	writer.write(invert)

	randomColour = 255 * np.array([rd.random(), rd.random(), rd.random()]) * np.ones(shape=(20,20,3), dtype='uint8')
	random_writer.write(randomColour.astype(dtype='uint8'))

	print(invert.shape)
	print(invert.dtype)

	cv.imshow('vid', invert)
	#can be changed to delay of 1 for live video
	if cv.waitKey(5) == ord('q'):
		break

capture.release()
random_writer.release()
writer.release()
cv.destroyAllWindows()
