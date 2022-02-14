import numpy as np
import constants as C
import math

# General class describing an aruco marker card. 
class Card:
	
	def __init__(self, fid, raw_corners):

		# Reshape corner array into a 4 x 2.
		corners = raw_corners.reshape((4, 2))#.astype(np.int32)

		# The centre coordinate of the aruco marker.
		self.position = 0.25 * corners.sum( axis=0 )

		# Vectors in the x and y direction of length 
		self.xvec = 0.25 * (corners[1] + corners[2] - corners[0] - corners[3])
		self.yvec = 0.25 * (corners[0] + corners[1] - corners[2] - corners[3])
		self.scale = (np.linalg.norm(self.xvec) + np.linalg.norm(self.yvec))
		#self.snap_distance = self.scale * 2
		
		# Rotation -pi -> +pi
		self.rotation = math.atan2(self.xvec[1], self.xvec[0])

		self.marker_corners = corners
		
		# Box or position where text/symbol goes.
		self.title_corners = np.zeros_like(corners)
		self.title_corners[0] = self.position - self.xvec + 2 * self.yvec
		self.title_corners[1] = self.position + self.xvec + 2 * self.yvec
		self.title_corners[2] = self.position + self.xvec + 4/3 * self.yvec
		self.title_corners[3] = self.position - self.xvec + 4/3 * self.yvec
		
		# Outer corners of paper card.
		self.outer_corners = np.zeros_like(corners)
		self.outer_corners[0] = self.position - 4/3 * self.xvec + 7/3 * self.yvec
		self.outer_corners[1] = self.position + 4/3 * self.xvec + 7/3 * self.yvec
		self.outer_corners[2] = self.position + 4/3 * self.xvec - 4/3 * self.yvec
		self.outer_corners[3] = self.position - 4/3 * self.xvec - 4/3 * self.yvec

		self.id = fid
		self.name = "Name"

	# # Draw function
	# def draw():
	# 	None
	# 	# Should draw the associated symbol or name of the card in the space at the top given for it.


