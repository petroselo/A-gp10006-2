# Inspired by splinefit of SA1 Project

import numpy as np
from scipy.interpolate import CubicSpline 

# knots[:,0], knots[:,1] are vectors giving the x and y locations of the knots.
def splinefit(knots, point_options=0):
	
	lengths = np.sqrt( (knots[1:,0] - knots[:-1,0])**2 + (knots[1:,1] - knots[:-1,1])**2 )
	# Integrate surface path length
	s = np.insert(np.cumsum(lengths), 0, 0)

	# Create cubic spline from surface length and knots
	spline = CubicSpline(s, knots)

	# Choose the samples along the surface
	if point_options == 0:
		return spline, None
	else:
		return spline, spline(np.linspace(0, s[-1], point_options))