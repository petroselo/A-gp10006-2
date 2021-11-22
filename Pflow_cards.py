import numpy as np
import constants as C

# pflow_card_dict = {
#     10:
# }

U = 1
m_over_2pi = 1
Γ_over_2pi = 1
μ_over_2pi = 1

class Pflow_card:
    def __init__(self, fid, corners, upsidedowncorners):
        self.pos = np.sum(corners, axis=0) / 4
        self.img_pos = np.sum(upsidedowncorners, axis=0) / 4
        self.img_scale = np.linalg.norm(upsidedowncorners[0] - upsidedowncorners[1])
        self.z0 = self.pos[0] + self.pos[1]*1j
        
        self.func = lambda z: np.zeros_like(z)

        # Flow
        if fid == 10:
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
