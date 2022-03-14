# Translated panel method from fortran.

# Change up from fortran which does 2 solves for a generalised 
# incident flow as we will always have only one incident flow

import numpy as np

# Panel: x,y of vorticesm
#      : U incident flow
#      : Boundary condition flag for free flow or turbine cascade. Could be the pitch as well.
#      : Output a vector of circulations or None if error.
def solve_panels(xv, yv, U, pitch=0):
	nv = len(xv)
	if nv != len(yv):
		return None

	# Set target points where BC are enforced and their surface-normal vectors.
	xt, yt, dl, nhatx, nhaty = target_points(xv, yv)

    # Normal flow through target points due to incident flow
	# U.nhat at each target point
	u_ind = U[0]*nhatx + U[1]*nhaty
	u_ind[-1] = 0

	# Calculate influence coefficients
	# inf[i,j]*Γ[j] is flow at target i due to vortex j normal to surface
	ainf = np.zeros((nv,nv))

	if pitch == 0: # Single aerofoil
		for ti in range(nv-1):
			for vi in range(nv):
				Δx = xt[ti] - xv[vi]
				Δy = yt[ti] - yv[vi]
				r2 = Δx*Δx + Δy*Δy
				ainf[ti, vi] = ( 1/(2*np.pi*r2) ) * (-nhatx[ti]*Δy  + nhaty[ti]*Δx )

	else: # Infinite Cascade
		pitch = BC_FLAG
		pass

    # Apply the kutta condition that vorticity at the final point = -(vorticity at the first point.) => no vorticity. 
	ainf[nv-1,0] = 1/dl[0]
	ainf[nv-1,nv-1] = 1/dl[nv-2]

     # Solve ainf*g=-u_ind for gam
	try:
		gam = np.linalg.solve(ainf, -u_ind)
		return gam
	except:
		return np.zeros(nv)


# The ith panel is always the next one along from the the ith point.
def target_points(xv, yv):
	#np.roll(a,-1) is equivalent to a[i+1].
	# ie indexing the next entry along
	xvp1 = np.roll(xv,-1)
	yvp1 = np.roll(yv,-1)

	# Target points are at the middle of each panel.
	xt = (xv + xvp1)/2
	yt = (yv + yvp1)/2

	δx = xvp1 - xv
	δy = yvp1 - yv
	dl = np.sqrt(δx*δx + δy*δy)

	nhatx = δy / dl
	nhaty = δx / dl

	return xt, yt, dl, nhatx, nhaty