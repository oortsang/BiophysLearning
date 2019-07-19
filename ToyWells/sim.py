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

# kB = 1.380649e-23 # units of J/K
class Particle():
    def __init__(self, Pot, pos = 0, DkT = 1, nsize = 1e-2, nsig = 1):
        self.Pot = Pot           # potential well
        self.F = -1*Pot.grad()   # force function
        self.DkT = DkT           # 1/gamma = D/kT
        self.nsize = nsize       # noise size
        self.nsig  = nsig        # stdev of gaussian noise
        self.dim = self.Pot.dim  # dimensionality
        self.pos = np.array(pos, dtype=np.float64) # ND position
        if self.pos.ndim == 0:
            self.pos = self.pos[np.newaxis]

        # could probably make this more efficient
        if (self.pos.shape[0] < self.dim):
            newpos = np.zeros(self.dim, dtype=np.float64)
            newpos[:self.pos.shape[0]] = self.pos
            self.pos = newpos
        # self.tot_noise = 0
        
    def step(self, dt):
        # pdb.set_trace()
        force = self.F(self.pos)
        noise = self.nsize * np.random.normal(0,self.nsig, self.dim)
        # self.tot_noise += noise
        self.v = self.DkT*force + noise
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
# CromWell next

# xs = np.arange(-2, 2, 0.05)
# Xs2d = np.transpose([np.tile(xs, xs.shape[0]), np.repeat(xs, len(xs))]).reshape((xs.shape[0], xs.shape[0], 2))
# xxs = Xs2d.reshape(Xs2d.shape[0]* Xs2d.shape[1], Xs2d.shape[2])
# outs = np.zeros(xxs.shape[0])
# for i in range(xxs.shape[0]):
#     outs[i] = TWell(xxs[i])
# outs = outs.reshape(Xs2d.shape[:2])
# outs = np.clip(outs, 0,5)
# plt.imshow(outs)
# plt.show()



WWell = MultiPolynomial([            0.5, 0, 0]) # wide
HWell = MultiPolynomial([             2, 0,  0]) # harmonic
DWell = 5e-2*MultiPolynomial([5,   0, -1, 0,  0]) # double
SWell = MultiPolynomial([1,        0, -3, 0,  0]) # shallow?
# NWell = MultiPolynomial([15, 0, -1, 0, -0.1, 0, 0])
# NWell = MultiPolynomial([5,0, -0.1, 0, 0])
OldNWell = MultiPolynomial([5,0, -0.1, 0, 0])
NWell = MultiPolynomial([80,0, -0.5, 0, 0]) # narrow (double) well

xs = np.arange(-0.1, 0.1, 0.01)
# plt.plot(xs, DWell(xs))
# plt.plot(xs, NWell(xs))
# plt.plot(xs, HWell(xs))
# plt.plot(xs, WWell(xs))
# # plt.plot(xs, DeepWells(xs))
# # plt.plot(xs, HWell(xs))
# plt.show()


# p1 = Particle(TWell, pos = np.array([0.5, -0.7]), nsize = 3, DkT = 0.1)
p2 = Particle(NWell, nsize = 0.9, DkT = 3.5, nsig = 1.5, pos = 0)
p3 = Particle(NWell, nsize = 0.5, DkT = 1, nsig = 3)
# p4 = Particle(NWell, nsize = 0.1, DkT = 1, nsig = 10)

ph1 = Particle(HWell, nsize = 0.1, DkT = 1, nsig = 6)
ph2 = Particle(WWell, nsize = 0.5, DkT = 2, nsig = 2)

# # Spit out the coordinates (and control the different trajectories...)

npart = 2
particles = [p2, ph1, p3, ph2,
             Particle(0.5*SWell, pos = 0.1, nsize = 5, DkT = 0.1),                 # 4
             Particle(MultiPolynomial([0.5, 1, 4]), pos = -2, nsize = 5e-1),       # 5
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
tstep = 2e-3
# nsteps = 105000
nsteps = 105000
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

if False:
    h5file = h5py.File('data/SimOutput.h5', 'w')
    h5file.create_dataset('particle_tracks', data=tracks)
    h5file.close()

# # To open:
# h5file = h5py.File('data/SimOutput.h5', 'r')
# b = h5file['particle_tracks'][:]
# h5file.close()

print("Fraction of time particle 1 spends on the right of 0:", np.sum(tracks[:,0] >= 0)/tracks.shape[0])

Hist, edges = np.histogram(tracks[:,0], bins = 30)
plt.plot(Hist)
plt.show()


plt.plot(tracks)
plt.show()
