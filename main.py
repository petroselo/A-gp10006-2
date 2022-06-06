#!/usr/bin/env python3

# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar, imshow
import io

#import matplotlib
import matplotlib.pyplot as plt # or could look into plotly or ggplot2
# from numpy.core.fromnumeric import shape

from splinefit import splinefit
from uniform_panels import uniform_points
from panel import solve_panels

DEBUG = True

# User-defined constants
import constants as C

# User modules
#from camera_calibration import calibrate
from perspective_calibration import get_table_camera_transform
from projector_calibration_manual import calibrate_projector

from Logic_cards import Logic_card
from Control_cards import Control_card
from Pflow_cards import Pflow_card
from Cards import Card
from Aerofoil import *

## Initial calibration values
cam_mtx = C.INITIAL_CALIBRATION_CM
dist_coeffs = C.INITIAL_CALIBRATION_DC

table_frame = np.empty(0)

def main():

	webcam, detect_params = initial_setup()

	cccc_code = cv.VideoWriter_fourcc(*'XVID')
	
	RECORDING = False

	CM = np.eye(3, dtype='float64')
	PM = np.eye(3, dtype='float64')
	table_dimensions = np.array([800, 600])

	projector_blank = False

	# The window that will be fullscreen on the projector.
	proj_window = cv.namedWindow(C.PROJ_WINDOW, cv.WND_PROP_FULLSCREEN)
	proj_img = np.zeros(shape=(C.PROJ_HEIGHT, C.PROJ_WIDTH, 3), dtype=np.uint8) + 255
	cv.putText(proj_img, 'Hello World', (960,540), C.FONT, 1, C.BLUE, 2)
	cv.imshow(C.PROJ_WINDOW, proj_img)

	table_overlay = np.zeros(shape=(table_dimensions[1]*C.TABLE_OVERLAY_FACTOR, table_dimensions[0]*C.TABLE_OVERLAY_FACTOR, 3), dtype=np.uint8)

## Main loop
	while True:

		ret, cam_frame = webcam.read()
		if not ret:
			print('Video finished')
			break

		table_frame = cv.warpPerspective(cam_frame, CM, table_dimensions)
	
		# Turn table frame into table overlay
		table_overlay *= 0
		process_frame(table_frame, detect_params, table_overlay, RECORDING)


		projector_output = cv.warpPerspective(table_overlay, PM, C.PROJ_SCREEN_DIMENSIONS)
		
		if RECORDING:
			projector_output[-3:-1,-3:-1,2] = 255
		cv.imshow(C.PROJ_WINDOW, (projector_output).astype(np.uint8))


		# Quitting condition
		inp = cv.waitKey(1)
		if inp == ord('q'):
			break

		# Generate, Save and Load previous perspective calibration
		if inp == ord('c'):
			CM, table_dimensions = get_table_camera_transform(C.BOARD6_2, webcam, detect_params, avg_frames=20,
																									cam_mtx=cam_mtx, dist_coeffs=dist_coeffs)
			table_overlay = np.zeros(shape=(table_dimensions[0]*C.TABLE_OVERLAY_FACTOR, table_dimensions[1]*C.TABLE_OVERLAY_FACTOR, 3), dtype=np.uint8)
			output = cv.VideoWriter('Output.mp4', cccc_code, 20.0, (table_dimensions[0], table_dimensions[1] ))
		
		if inp == ord('l'):
			CM = np.loadtxt('PerspectiveMatrix.txt')
			table_dimensions = np.loadtxt('Dimensions.txt', dtype='int64')
			table_overlay = np.zeros(shape=(table_dimensions[1]*C.TABLE_OVERLAY_FACTOR, table_dimensions[0]*C.TABLE_OVERLAY_FACTOR, 3), dtype=np.uint8) + 100
			output = cv.VideoWriter('Output.mp4', cccc_code, 20.0, (table_dimensions[0], table_dimensions[1] ))
		if inp == ord('s'):
			np.savetxt('PerspectiveMatrix.txt', CM)
			np.savetxt('Dimensions.txt', table_dimensions, fmt='%u')
		
		#Toggle projector window to fullscreen.
		if inp ==ord('f'):
			if cv.getWindowProperty(C.PROJ_WINDOW, cv.WND_PROP_FULLSCREEN) == cv.WINDOW_NORMAL:
				cv.setWindowProperty(C.PROJ_WINDOW,cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
			else:
				cv.setWindowProperty(C.PROJ_WINDOW,cv.WND_PROP_FULLSCREEN,cv.WINDOW_NORMAL)

		# Projector calibration
		if inp == ord('p'):
			PM = calibrate_projector(webcam, CM, table_dimensions, avg_frames=20, detect_params=detect_params)
			print(PM)
		if inp == ord('r'):
			RECORDING = not RECORDING
			
		if inp == ord('b'):
			if projector_blank:
				imshow(C.PROJ_WINDOW, np.zeros(shape=C.PROJ_SHAPE))
			else:
				imshow(C.PROJ_WINDOW, np.zeros(shape=C.PROJ_SHAPE))
			projector_blank = not projector_blank

	webcam.release()
	output.release()
	cv.destroyAllWindows()

def draw_line(img, start, end, colour):
	# Draw a line on the table frame.q
	cv.line(img, tuple(start.astype(np.int32)), tuple(end.astype(np.int32)), colour)
	# Draw a corresponding line on the output
	#cv.line(table_frame, tuple(start.astype(np.int32)), tuple(end.astype(np.int32)))

def initial_setup():

	if DEBUG: print('Starting initial setup...')

	detect_params = cv.aruco.DetectorParameters_create()

	# Set up webcam
	webcam = cv.VideoCapture(1)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, 800)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
	#webcam.set(cv.CAP_PROP_FPS, 25)
	if DEBUG: print("Check 1")

	webcam.set(cv.CAP_PROP_AUTOFOCUS, 0) 	#Disable autofocus
	if not webcam.isOpened():
		print('Failed to open camera.')
		exit()

	if DEBUG: print('Finish initial setup...')

	return webcam, detect_params

def process_frame(table_frame, detect_params, table_overlay, RECORDING):
	logic_cards = []
	pflow_cards = []
	aerofoil_cards = []
	cards = []

	(allCorners, ids, rejected) = cv.aruco.detectMarkers(table_frame, C.DICTIONARY, parameters = detect_params)

		# Construct list of logic cards and their positions.

		#send that information off to another function to analyse the logic and work out what to do about it and draw that on the frame.

	if len(allCorners) > 0:
		fids = ids.flatten()
		markers = zip(fids, allCorners)
		for (fid, raw_corners) in markers:

			# Add to general card list.
			if fid >= 30 and fid < 40:
				cards.append( Card(fid, raw_corners) )
					

			# If in logic card range of markers.
			if fid > 19 and fid < 28:
				corners = raw_corners.reshape((4, 2)).astype(np.int32)
				logic_cards.append( Logic_card(fid, corners))

			# If in pflow card range
			# if fid > 9 and fid < 16:
			# 	img_corners = raw_corners.reshape((4, 2)).astype(np.int32)
			# 	corners = (raw_corners.reshape((4, 2)).astype(np.int32) * np.array([1, -1]) + np.array([0, table_dimensions[1]]) )/20 # flip the y coord.
			# 	pflow_cards.append( Pflow_card(fid, corners, img_corners))

			# If in control range.
			# if fid >= 30 and fid <= 37:
			# 	corners = raw_corners.reshape((4, 2)).astype(np.int32)
			# 	control_cards.append( Control_card(fid, corners))

			# Aerofoil cards
			if fid >= 40 and fid < 50:
				corners = raw_corners.reshape((4, 2)).astype(np.int32)
				# Don't add repeated cards.
				if not any(fid == card.id for card in aerofoil_cards):
					aerofoil_cards.append( Point_card(fid, corners))

	# Turn Aerofoil cards list into a list of spline points.
	# Order by id
	if len(aerofoil_cards) > 1:
		aerofoil_cards.sort(key = lambda a:a.id)
		# Set up the spline knots and repeat the first point at the end
		knot_points = [card.point for card in aerofoil_cards]
		knot_points.append(knot_points[0])
		knots = np.array(knot_points) # n x 2
		
		_, visual_spline_points = splinefit(knots)

		# Convert knots to uniform spaced vortices. There are n-1 panels as two vortices at trailing edge.
		n_vort = 100
		_, hi_res_spline_points = splinefit(knots, 5*n_vort)
		vortex_points = uniform_points(hi_res_spline_points, n_vort)

		U = np.array([1,0])
		
		# Vortex circulations
		gam = solve_panels(vortex_points[:,0], vortex_points[:,1], U, BC_FLAG=0)



		# Draw the points and connecting lines and the splines.
		for card in aerofoil_cards:
			card.draw(table_frame, table_overlay, C.TABLE_OVERLAY_FACTOR)

		cv.polylines(table_frame, [visual_spline_points.astype(np.int32)], isClosed=False, color=C.WHITE)
		#cv.polylines(table_frame, [knots.astype(np.int32)], isClosed=False, color=C.RED)

		cv.polylines(table_overlay, [(C.TABLE_OVERLAY_FACTOR*visual_spline_points).astype(np.int32)], isClosed=False, color=C.WHITE)
		#cv.polylines(table_overlay, [(C.TABLE_OVERLAY_FACTOR*knots).astype(np.int32)], isClosed=False, color=C.RED)

		aerofoil_cards.clear()

		#spline



		 


	# for card in cards:
	# 	for i in range(4):
	# 		draw_line(table_frame, card.outer_corners[i], card.outer_corners[(i+1)%4], C.RED)
	# 		draw_line(table_frame, card.title_corners[i], card.title_corners[(i+1)%4], C.BLUE)
	# 		cv.putText(table_frame, f'{card.rotation*180/np.pi}', tuple(card.title_corners[3].astype(np.int32)), C.FONT, C.FONT_WIDTH, C.BLUE, 2)

	# cards.clear()

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

			cv.line(table_overlay, tuple(2*inp.pos.astype('int')), tuple(2*(inp.pos+lc.xvec).astype('int')), C.BLUE, 2)
			cv.circle(table_overlay, tuple(2*inp.pos.astype('int')), int(0.5*lc.scale), C.BLUE)
			if inp.conn is not None:
				cv.line(table_overlay, tuple(2*inp.pos.astype('int')), tuple(2*(inp.conn.pos).astype('int')), C.RED, 2)

		for o in lc.outps:
			cv.circle(table_frame, tuple(o.pos.astype('int')), int(0.25*lc.scale), C.BLUE if o.val>1 else (C.GREEN if o.val==1 else C.RED), -1 if o.val<2 else 1)
			cv.line(table_frame, tuple(o.pos.astype('int')), tuple((o.pos-lc.xvec).astype('int')), C.BLUE, 2 if o.val<2 else 1)

			cv.circle(table_overlay, tuple(2*o.pos.astype('int')), int(0.5*lc.scale), C.BLUE if o.val>1 else (C.GREEN if o.val==1 else C.RED), -1 if o.val<2 else 1)
			cv.line(table_overlay, tuple(2*o.pos.astype('int')), tuple(2*(o.pos-lc.xvec).astype('int')), C.BLUE, 4 if o.val<2 else 2)

	logic_cards.clear()

	if len(pflow_cards) > 0:
		# Construct a grid.
		xvec, yvec = np.linspace(0, table_dimensions[0]/20, 101), np.linspace(0, table_dimensions[1]/20, 101)
		x, y = np.meshgrid(xvec, yvec, indexing='ij')
		z = x + y*1j

		#print(x[:3,:3])
		F = np.zeros_like(z) # = φ + jψ
		# Add all the card functions to the grid
		for card in pflow_cards:
			F += card.F(z)

		ψ = np.imag(F)

		p = None
		if 15 in fids:
			#p = p0 - .5ρ(u**2 + v**2) =  - (dψ/δy)^2 -(dψ/dx)^2
			dpsibydy = (ψ[1:-1,:-2]-ψ[1:-1,2:])/(y[1:-1,:-2]-y[1:-1,2:])
			dpsibydx = (ψ[:-2,1:-1]-ψ[2:,1:-1])/(x[:-2,1:-1]-x[2:,1:-1])
			p = -  ( dpsibydy )**2 - ( dpsibydx)**2

			for card in pflow_cards:
				mask = ((x[1:-1,1:-1]-card.pos[0])**2 + (y[1:-1,1:-1]-card.pos[1])**2) > card.scale2
				p = mask * p + np.max(p)

		# Plot the contours, exclude cards and draw on the table overlay.
		fig = plt.figure(dpi=100, figsize=(table_dimensions[0]/100, table_dimensions[1]/100), frameon=False ) #facecolor should be irrelevant.
		ax = fig.add_subplot(111)
		#ax.axis('off')

		ax.set_facecolor('black')
		ax.contour(x, y, ψ, colors="red", levels=np.linspace(np.min(ψ), np.max(ψ), 23), antialiased=False, linestyles='solid') # BGR / RGB switching means Red <-> Blue
		#ax.contourf(x, y, ψ, levels=23, antialiased=False, linestyles='solid') # BGR / RGB switching means Red <-> Blue
		
		if not p is None:
			ax.contourf(x[1:-1, 1:-1], y[1:-1, 1:-1], p, levels=23)

		
		fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
		fig.canvas.draw()

		buf = io.BytesIO()
		fig.savefig(buf, format='raw', dpi=100*C.TABLE_OVERLAY_FACTOR , bbox_inches=0,pad_inches = 0)

		plt.close(fig)
		buf.seek(0)
		plot_img = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
					newshape=(int(fig.bbox.bounds[3])*C.TABLE_OVERLAY_FACTOR, int(fig.bbox.bounds[2])*C.TABLE_OVERLAY_FACTOR, -1))[:,:,:3] #chop off alpha layer by taking only top 3 RGB layers
		buf.close()

		mask = 255 - cv.inRange(plot_img, C.BLACK, C.BLACK)
		# Exclude card location them selves.

		#cv.imshow('Plot image', plot_img)
		#cv.imshow('mask', mask)
		cv.copyTo(plot_img, mask, table_overlay)
		#table_overlay = table_overlay // 1.2

		# Don't project over cards themselves.
		for card in pflow_cards:
			cv.circle(table_overlay, (C.TABLE_OVERLAY_FACTOR*card.img_pos).astype(int), int(C.TABLE_OVERLAY_FACTOR*card.img_scale), C.BLACK, -1)

		pflow_cards.clear()

	# if len(control_cards) > 0:
	# 	for cc in control_cards:
	# 		for inp in cc.inps:
	# 			for ccc in logic_cards:
	# 				if cc is not ccc: # exclude current card
	# 					for outp in ccc.outps:
	# 						dist = np.linalg.norm(outp.pos - inp.pos)
	# 						if dist < lc.snap_distance and ( (inp.conn is None) or (dist < np.linalg.norm(inp.pos - inp.conn.pos) )):
	# 							inp.conn = outp

	# if len(control_cards) > 0:
	# 	for cc in control_cards:
	# 		#if cc.fid == 31:

	# 		if cc.fid == 33:
	# 			cv.circle(table_overlay, tuple(2*cc.position.astype('int')), int(3*cc.scale), C.BLUE)
	# 			cv.putText(table_overlay, f'Value: {cc.rot}', tuple(2*cc.position.astype('int')), C.FONT, 1, C.BLUE, 1, lineType=cv.LINE_AA)

	# 	control_cards.clear()

	cv.imshow(C.VIDEO_TITLE, table_frame)

	# # Save video
	# if RECORDING:
	# 	output.write(table_frame)

	return table_overlay

if __name__ == "__main__":
	main()
