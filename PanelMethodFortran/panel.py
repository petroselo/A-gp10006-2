# Translated panel method from fortran.

import numpy as np

# Panel: x,y of vortices
#      : U is the incident flow
#      : Boundary condition flag for free flow or turbine cascade. Could be the pitch as well.
#      : Output a vector of circulations
def panel(xv, yv, U, BC_FLAG):
	
	# Set target points where BC are enforced and their surface-normal vectors.
	xt, yt, dl, nhatx, nhaty = target_points(xv, yv)

    # Flow through target points due to incident flow
	# U.nhat at each target point
	u_ind = U[0]*nhatx + U[1]*nhaty
	u_ind[-1] = 0

	# Calculate influence coefficients
	# inf[i,j]*Γ[j] is flow at target i due to vortex j normal to surface
	if BC_FLAG == 0: # Free flow

	else: # Infinite Cascade


      call apply_kutta

      call invert(gamindx)

	# Change up from fortran which does 2 solves for a generalised 
	# incident flow as we will always have only one incident flow

	


# The ith panel is always the next one along from the the ith point.
def target_points(xv, yv):
	#np.roll(a,-1) is equivalent to a[i+1].
	# ie indexing the next entry along
	xvp1 = np.roll(xv,-1)
	yvp1 = np.roll(yv,-1)

	# Target points are at the middle of each panel.
	xt = (xv + xvp1 )/2
	yt = (yv + yvp1 )/2

	δx = xvp1 - xv
	δy = yvp1 - yv
	dl = np.sqrt(δx*δx + δy*δy)

	nhatx = δy / dl
	nhaty = δx / dl

	return xt, yt, dl, nhatx, nhaty