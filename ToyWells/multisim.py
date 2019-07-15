import numpy as np
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

from multipoly import MultiPolynomial

# FF = MultiPolynomial([[13,1,-4,4],[0,-2,0,3]])
# FF = MultiPolynomial([[0,0,1], [0,1,2],[1,-2, 4]])
# GG = FF.grad()


# Brownian motion / taking random steps

kB = 1.380649e-23 # units of J/K
class Particle():
    def __init__(self, Pot, pos = 0, DkT = 1, nsize = 1e-2):
        self.Pot = Pot           # potential well
        self.F = -1*Pot.grad()   # force function
        self.DkT = DkT           # 1/gamma = D/kT
        self.nsize = nsize       # noise size
        self.dim = self.Pot.dim    # dimensionality
        self.pos = np.array(pos, dtype=np.float64) # ND position
        if self.pos.ndim == 0:
            self.pos = self.pos[np.newaxis]

        # could make this more efficient but I'm not in the mood...
        # if (self.pos.ndim == 0 and self.dim > 1):
        #     newpos = np.zeros(self.dim, dtype=np.float64)
        #     newpos[0] = self.pos
        #     self.pos = newpos
        if (self.pos.shape[0] < self.dim):
            newpos = np.zeros(self.dim, dtype=np.float64)
            newpos[:self.pos.shape[0]] = self.pos
            self.pos = newpos

    def step(self, dt):
        # pdb.set_trace()
        force = self.F(self.pos)
        noise = self.nsize * np.random.normal(0,1, self.dim)
        self.v = self.DkT*force + noise
        # pdb.set_trace()
        self.pos += dt * self.v

# QuadWell = MultiPolynomial([[1,0,-6,0,0],[1,-0.5,-1,1,0]]) # 2 dimensions!

TriCoeffs = np.array([ [0, 0, 0, 0, 0, 0,   1],   # x^6
                       [0, 0, 0, 0, 0, 0, 0.2],     # x^5
                       [0,0,0,0,3, 0.02, -0.46],      # x^4
                       [0,0,0,0,0.4, 0.12, -1.762],     # x^3
                       [0,0,3,0.4,0.04, -0.726, -0.1395], # x^2
                       [0,0,0.2,0.12,5.118, 0.292, 0.405],  # x^1
                       [1, 0.2, .5, .314, .2565, .114, .7565] # x^0
                     ])
QCoeffs = np.zeros((5,5))
QCoeffs[4,4] = -1
QCoeffs[4,2] = 1
QCoeffs[2,4] = 1
QCoeffs[2,2] = -1


TWell = MultiPolynomial(TriCoeffs)
QWell = MultiPolynomial(QCoeffs)


HWell = MultiPolynomial([              1, 0, 0])
DWell = 5e-2*MultiPolynomial([5,   0, -1, 0, 0])
SWell = MultiPolynomial([1,        0, -3, 0, 0])


# CromWell next

p1 = Particle(TWell, pos = np.array([0.5, -0.7]), nsize = 3, DkT = 0.1)
p2 = Particle(MultiPolynomial([0.5, 1, 4]), pos = -2, nsize = 5e-1)
p3 = Particle(SWell, pos = 0.1, nsize = 5, DkT = 0.1)


# xs = np.arange(-3, 3, 0.05)
# plt.plot(xs, TWell(xs))
# plt.plot(xs, DWell(xs))
# plt.plot(xs, SWell(xs))
# plt.show()

xs = np.arange(-2, 2, 0.05)
Xs2d = np.transpose([np.tile(xs, xs.shape[0]), np.repeat(xs, len(xs))]).reshape((xs.shape[0], xs.shape[0], 2))
xxs = Xs2d.reshape(Xs2d.shape[0]* Xs2d.shape[1], Xs2d.shape[2])
outs = np.zeros(xxs.shape[0])
for i in range(xxs.shape[0]):
    outs[i] = TWell(xxs[i])
outs = outs.reshape(Xs2d.shape[:2])
outs = np.clip(outs, 0,5)
plt.imshow(outs)
# plt.imshow(QWell(Xs2d.flatten()).reshape(Xs2d.shape))
plt.show()

# # Spit out the coordinates (and control the different trajectories...)

npart = 4
particles = [p1, p2, p3,
             Particle(2*DWell, nsize = 0.5, DkT = 0.5),                            # 4
             Particle(MultiPolynomial([0.01, 0, -1]), pos = 0, nsize = 1),         # 5 # moves around a lot
             Particle(MultiPolynomial([0.5, 1, 4]), pos = -2, nsize = 1),          # 6
             Particle(MultiPolynomial([2, 13, 0]), pos = -0.1, nsize = 1),         # 7
             Particle(DWell,                        pos = 0, nsize = 1e-1),        # 8
             Particle(MultiPolynomial([0.5, 1, -1]), pos = 0.2, nsize = 1e-1),     # 9
             Particle(MultiPolynomial([0.5, 1, 1]), pos = -0.3, nsize = 1e-1),     # 10
             Particle(MultiPolynomial([0.7, 1, 1]), nsize = 1e-1, DkT = 1e-4),     # 11
             Particle(MultiPolynomial([0.5, 1, 1]), pos = 0, nsize = 1, DkT = 10), # 12
             Particle(MultiPolynomial([3, 1, 1]), pos = 0, nsize = 1),             # 13
             Particle(MultiPolynomial([10, -10, -1]), pos = 0, nsize = 1),         # 14
             Particle(MultiPolynomial([0.5, 1, 1]), pos = -0.1, nsize = 1),        # 15
             # Particle(MultiPolynomial([0.5, 1, -1]), pos = 0.5, nsize = 1)       # 16
            ][:npart]
tstep = 1e-2
nsteps = 100000
dimensions = 0
for i in range(npart):
    dimensions += particles[i].dim
print("Dimensions:", dimensions)    
tracks = np.zeros((nsteps, dimensions), dtype=np.float32)
for i in tqdm(range(tracks.shape[0])):
    j = 0
    for p in particles:
        # pdb.set_trace()
        p.step(tstep)
        pdim = p.dim
        tracks[i,j:j+pdim] = p.pos
        j+= pdim

if True:
    h5file = h5py.File('data/SimOutput.h5', 'w')
    h5file.create_dataset('particle_tracks', data=tracks)
    h5file.close()

# # To open:
# h5file = h5py.File('data/SimOutput.h5', 'r')
# b = h5file['particle_tracks'][:]
# h5file.close()


plt.plot(tracks)
plt.show()
