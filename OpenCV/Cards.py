import numpy as np
import constants as C

# General class describing an aruco marker card. 
class Card:
	
	def __init__(self, fid, corners):
				
		# The centre coordinate of the aruco marker.
		self.position = 0.25 * corners.sum( axis=0 )

		# Vectors in the x and y direction of length 
		self.xvec = 0.25 * (corners[1] + corners[2] - corners[0] - corners[3])
		self.yvec = 0.25 * (corners[0] + corners[1] - corners[2] - corners[3])
		self.scale = (np.linalg.norm(self.xvec) + np.linalg.norm(self.yvec))
		self.snap_distance = self.scale * 2

		self.marker_corners = corners
		# box or position where text/symbol goes.
		
		# Outer corners of paper card.
		self.outer_corners = np.zeros_like(corners)
		self.outer_corners[0] = self.position - self.yvec + 7/3 * self.xvec
		self.outer_corners[1] = self.position + self.yvec + 7/3 * self.xvec
		self.outer_corners[2] = self.position + 7/3 * self.scale
		self.outer_corners[3] = self.position + 7/3 * self.scale
