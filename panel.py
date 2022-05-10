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

	print(dl)

    # Normal flow through target points due to incident flow
	# U.nhat at each target point
	u_ind = np.zeros(nv)
	u_ind[0:nv-1] = U[0]*nhatx + U[1]*nhaty
	u_ind[nv-1] = 0
	print(u_ind)


	# Calculate influence coefficients
	# inf[i,j]*Γ[j] is flow at target i due to vortex j normal to surface
	ainf = np.zeros((nv,nv))

	if pitch == 0:              # Single aerofoil
		for ti in range(nv-1):
			for vi in range(nv):
				Δx = xt[ti] - xv[vi]
				Δy = yt[ti] - yv[vi]
				r2 = Δx*Δx + Δy*Δy
				ainf[ti, vi] = ( 1/(2*np.pi*r2) ) * (-nhatx[ti]*Δy  + nhaty[ti]*Δx )

	else:                       # Infinite Cascade
		for ti in range(nv-1):
			for vi in range(nv):
				x = (2*np.pi * ( xt[ti]-xv[vi] ) )/pitch
				y = (2*np.pi * ( yt[ti]-yv[vi] ) )/pitch
				ainf[ti,vi] = -(nhatx[ti]*(1/(2*pitch))*np.sin(y))/(np.cosh(x)-np.cos(y))+(nhaty[ti]*(1/(2*pitch))*np.sinh(x))/(np.cosh(x)-np.cos(y))

    # Apply the kutta condition that vorticity at the final point = -(vorticity at the first point.) => no vorticity, smooth flow
	ainf[nv-1, 0] = 1/(dl[0])
	ainf[nv-1, nv-1] = 1/(dl[nv-2])

	print(ainf)
     # Solve ainf*g=-u_ind for gam
	try:
		gam = np.linalg.solve(ainf, -u_ind)
		return gam, xt, yt, nhatx, nhaty #todo remove extra returns
	except:
		return np.zeros(nv)


# The ith panel is always the next one along from the the ith point.
def target_points(xv, yv):

	# Target points are at the middle of each panel.
	xt = (xv[0:-1] + xv[1:])/2
	yt = (yv[0:-1] + yv[1:])/2

	δx = xv[1:] - xv[0:-1]
	δy = yv[1:] - yv[0:-1]
	dl = np.sqrt(δx*δx + δy*δy)

	nhatx = δy / dl
	nhaty = -δx / dl

	return xt, yt, dl, nhatx, nhaty