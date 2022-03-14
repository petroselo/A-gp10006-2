from random import sample
import numpy as np


# Convert a (nx2) array of x,y points into n uniform spaced vortices
def uniform_points(hi_res_spline_points, nv):

	p = hi_res_spline_points

	# Distance along the surface of each hi_res_point 
	panel_lengths = np.sqrt( (p[1:,0]-p[0:-1,0])**2 + (p[1:,1]-p[0:-1,1])**2 )
	surface_integral = np.insert( np.cumsum(panel_lengths), 0, 0)

	surface_length = surface_integral[-1]

	# Determine length of a uniform panel
	spacing = surface_length / (nv-1)

	sample_locations = np.linspace(0, surface_length, nv)

	vortex_points = np.zeros((nv,2))
	vortex_points[:,0] = np.interp(sample_locations, surface_integral, p[:,0])
	vortex_points[:,1] = np.interp(sample_locations, surface_integral, p[:,1])

	np.interp


	return vortex_points