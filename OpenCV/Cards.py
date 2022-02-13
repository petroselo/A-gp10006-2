import numpy as np
import constants as C

# General class describing an aruco marker card. 
class Card:
	
	def __init__(self, fid, corners):
		
		self.corners = corners
		
		# The centre coordinate of the aruco marker.
		self.position = 0.25 * corners.sum( axis=0 )

		# Vectors in the x and y direction of length 
		self.xvec = 0.25 * (corners[1] + corners[2] - corners[0] - corners[3])
		self.yvec = 0.25 * (corners[0] + corners[1] - corners[2] - corners[3])
		self.scale = (np.linalg.norm(self.xvec) + np.linalg.norm(self.yvec))
		self.snap_distance = self.scale * 2