import numpy as np
import constants as C

# pflow_card_dict = {
#     10:
# }

U = 0.25
m_over_2pi = 1
Γ_over_2pi = 1
μ_over_2pi = -2

class Pflow_card:
    def __init__(self, fid, corners, img_corners):
        self.pos = np.sum(corners, axis=0) / 4
        self.img_corners = img_corners
        self.img_pos = np.sum(img_corners, axis=0) / 4
        self.img_scale = 0.9 * np.linalg.norm(img_corners[0] - img_corners[1])
        self.scale2= (0.7 * np.linalg.norm(self.pos-corners[0]) )**2
        self.z0 = self.pos[0] + self.pos[1]*1j
        
        #self.func = lambda z: np.zeros_like(z)

        # Flow
        if fid == 10:
           # a = atan...
            self.F = lambda z: U*z # no angles yet :(
        # Source
        elif fid == 11:
            self.F = lambda z: m_over_2pi * np.log(z-self.z0)

        # Sink
        elif fid == 12:
            self.F = lambda z: -m_over_2pi * np.log(z-self.z0)

        # Doublet
        elif fid == 13:
            self.F = lambda z: - μ_over_2pi / (z - self.z0) # no angles yet :(

        # Vortex
        elif fid == 14:
            self.F = lambda z: -1j*Γ_over_2pi * np.log(z-self.z0)
        else:
            self.F = lambda z: 0
