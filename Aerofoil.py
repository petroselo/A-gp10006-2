import numpy as np
import cv2 as cv

from Cards import *

class Point_card(Card):
	def __init__(self, fid, raw_corners):
		super().__init__(fid, raw_corners)
		self.point = self.position -1.5 * self.yvec

	def draw(self, table, overlay, factor):
		# Draw a blue circle under the card indicating the point.
		cv.circle(table, tuple(self.point.astype(np.int32)), int(0.1*self.scale), C.BLUE)
		cv.circle(overlay, tuple((factor*self.point).astype(np.int32)), int(factor*0.1*self.scale), C.BLUE)



