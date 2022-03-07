# External Libraries
import numpy as np
import cv2 as cv
from cv2 import aruco as ar
from fpdf import FPDF
import os


pdf = FPDF(orientation = 'L', unit = 'mm', format = 'A4')
pdf.add_page()

wid = 297
hei = 210

#pdf.set_font('helvetica', 'bold', 10)
pdf.set_text_color(255, 255, 255)

dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
border = 1
first_id = 40
tag_width = dictionary.markerSize + 2*border

pdf.line(0, 105, wid, 105)
for i in range(1,4):
	pdf.line(i*wid/4, 0, i*wid/4,hei)

for i in range(8):
	tag_id = first_id + i
	# Generate marker
	fiducial = np.zeros((tag_width,tag_width,1), dtype='uint8')
	cv.aruco.drawMarker(dictionary, tag_id, tag_width, fiducial, border)
	cv.imwrite(f'temp{i}.png', fiducial)
	# Place in pdf
	x = i % 4
	y = 1 if i > 3 else 0
	pdf.image(f'temp{i}.png', x = x*wid/4 + 7.125 , y = y * hei / 2 + 105 - 60 -7.125, w = 60, h = 60)
	# Could add text here


pdf.output('aruco_cards.pdf')

cv.waitKey(20)

for i in range(8):
	os.remove(f'temp{i}.png')
