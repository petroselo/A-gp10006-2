import numpy as np
import cv2 as cv

tagDict = cv.aruco.Dictionary_get( cv.aruco.DICT_4X4_50 )
detectParams = cv.aruco.DetectorParameters_create()

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
TEAL = (255, 255, 0)
width = 2

menuRadius = 100

DEBUGid = 0
DEBUGrejectedId = 1

webcam = cv.VideoCapture(0)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, 800) #800
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 600) #600
#webcam.set(cv.CAP_PROP_FPS, 25)


if not webcam.isOpened():
	print('Failed to open camera.')
	exit()


while True:

	DEBUG = False
	DEBUGrEJECTED = False

	ret, frame = webcam.read()
	if not ret:
		print('Video finished')
		break

	(allCorners, ids, rejected) = cv.aruco.detectMarkers(frame, tagDict, parameters = detectParams)

	if len(allCorners) > 0:
		fids = ids.flatten()
		markers = list(zip(fids, allCorners))


		if DEBUGid in fids:
			DEBUG = True
			debugMarker = next(m for m in markers if m[0] == DEBUGid)
			(dbtl, dbtr, dbbr, dbbl) = debugMarker[1].reshape((4, 2))
			debugX = int( (dbtl[0] + dbbr[0]) / 2.0 )
			debugY = int( (dbtl[1] + dbbr[1]) / 2.0 )
			cv.circle(frame, (debugX, debugY), menuRadius, BLUE)

			# Further debug options
			if DEBUGrejectedId in fids:
				debugrMarker = next(m for m in markers if m[0] == DEBUGrejectedId)
				(dbrtl, dbrtr, dbrbr, dbrbl) = debugrMarker[1].reshape((4, 2))
				debugrX = int( (dbrtl[0] + dbrbr[0]) / 2.0 )
				debugrY = int( (dbrtl[1] + dbrbr[1]) / 2.0 )
				if (debugrX - debugX)**2 + (debugrY - debugY)**2 < menuRadius**2:
					DEBUGrEJECTED = True

#		if 10 in markers and 11 in markers and 12 in markers and 13 in markers:
#			(tl, tr, bl, br) =

		for (fid, tagCorners) in markers:
			# Convert to ints in the frame.
			(tl, tr, br, bl) = pts = tagCorners.reshape((4, 2)).astype(np.int32)


			#Draw lines to show markers and rejected options.
			if DEBUG:
				cv.polylines(frame, [pts], True, GREEN, width)

	if DEBUGrEJECTED:
		for reject in rejected:
			pts = reject.reshape((4,2)).astype(np.int32)
			cv.polylines(frame, [pts], True, RED, width)
			cv.line(frame, pts[0], pts[2],RED, width)
			cv.line(frame, pts[1], pts[3],RED, width)


	cv.imshow('Video', frame)
	if cv.waitKey(1) == ord('q'):
		print(frame.shape)
		break


webcam.release()
cv.destroyAllWindows()
