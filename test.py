import numpy as np
import matplotlib.pyplot as plt

from splinefit import splinefit
from uniform_panels import uniform_points
from panel import solve_panels

knot_points = [ [600,300], [500,350], [300,350], [250,285], [300,300], [500,300] ]
knot_points.append(knot_points[0])
knots = np.array(knot_points) + 0.0002 # n x 2

_, visual_spline_points = splinefit(knots, 200)

# Convert knots to uniform spaced vortices. There are n-1 panels as two vortices at trailing edge.
n_vort = 100
_, hi_res_spline_points = splinefit(knots, 5*n_vort)

vortex_points = uniform_points(hi_res_spline_points, n_vort)

#print(vortex_points) - we know first and last point are the same.
#print(len(vortex_points)-n_vort) = 0

U = np.array([1,0])

# Vortex circulations
gam, xt, yt, nhx, nhy = solve_panels(vortex_points[:,0], vortex_points[:,1], U, pitch=200)

print("Gamma sum = ", np.sum(gam))

if gam is None or (all(gam == 0)):
	print("Solve solution failed")

xvec, yvec = np.linspace(-8000, 8000, 200), np.linspace(-6000, 6000, 200)
x, y = np.meshgrid(xvec, yvec, indexing='ij')
z = x + y*1j
F = np.zeros_like(z) # = φ + jψ

#Add incident flow
F += U[0] * z

# Add vortices
for i in range(n_vort):
	dz = z - (vortex_points[i,0]+vortex_points[i,1]*1j)
	F += -1j*gam[i]/(2*np.pi) * np.log(dz)

ψ = np.imag(F)

plt.plot(visual_spline_points[:,0], visual_spline_points[:,1])
plt.plot(knots[:,0], knots[:,1], 'or')
plt.plot(vortex_points[:,0], vortex_points[:,1], 'ok')
plt.plot(xt, yt, 'og')
plt.plot(xt+nhx, yt+nhy, 'om')
plt.contour(x, y, ψ, levels=np.linspace(np.min(ψ), np.max(ψ), 80), antialiased=False, linestyles='solid')
plt.show()
plt.axes().set_aspect('equal')