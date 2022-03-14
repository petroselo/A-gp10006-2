import numpy as np
import matplotlib.pyplot as plt

from splinefit import splinefit
from uniform_panels import uniform_points
from panel import solve_panels

knot_points = [ [600,300], [500,350], [300,350], [250,285], [300,300], [500,300] ]
knot_points.append(knot_points[0])
knots = np.array(knot_points) + 0.0002# n x 2

_, visual_spline_points = splinefit(knots, 100)

# Convert knots to uniform spaced vortices. There are n-1 panels as two vortices at trailing edge.
n_vort = 400
_, hi_res_spline_points = splinefit(knots, 5*n_vort)
vortex_points = uniform_points(hi_res_spline_points, n_vort)

print(vortex_points)

U = np.array([1,0])

# Vortex circulations
gam = solve_panels(vortex_points[:,0], vortex_points[:,1], U)

#print(gam)

xvec, yvec = np.linspace(0, 800, 101), np.linspace(0, 600, 101)
x, y = np.meshgrid(xvec, yvec, indexing='ij')
z = x + y*1j
F = np.zeros_like(z) # = φ + jψ

#Add incident flow +vortices  
F += z
for i in range(n_vort):
	dz = z - (vortex_points[i,0]+vortex_points[i,1]*1j)
	F += 1j*gam[i]/(2*np.pi) * np.log(dz)

ψ = np.imag(F)

plt.plot(visual_spline_points[:,0], visual_spline_points[:,1])
plt.plot(knots[:,0], knots[:,1], 'or')
plt.plot(vortex_points[:,0], vortex_points[:,1], 'ok', scalex=0.001, scaley=0.001)
plt.contourf(x, y, ψ, levels=np.linspace(np.min(ψ), np.max(ψ), 40), antialiased=False, linestyles='solid')
plt.show()