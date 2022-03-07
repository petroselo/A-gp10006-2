# Inspired by splinefit of SA1 Project

import numpy as np
from scipy.interpolate import CubicSpline 

# knots[:,0], knots[:,1] are vectors giving the x and y locations of the knots.
def splinefit(knots, point_options=0):
	
	lengths = np.sqrt( (knots[1:,0] - knots[:-1,0])**2 + (knots[1:,1] - knots[:-1,1])**2 )
	# Surface path length
	s = np.insert(np.cumsum(lengths), 0, 0)

	# Create cubic spline from surface length and knots
	spline = CubicSpline(s, knots)

	# Choose the samples along the surface

	# Default, just choose 1000 points to complete the curve.
	if point_options == 0:
		samples = np.linspace(0, s[-1], 1000)

	# Return the resulting
	return spline(samples)
	
	return spline
