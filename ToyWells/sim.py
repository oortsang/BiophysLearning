import numpy as np
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import pdb

from multipoly import MultiPolynomial

# Brownian motion / taking random steps

# kB = 1.380649e-23 # units of J/K
# m / s * (a / m) * (s / ps)
#         (1e-10)    (1e12)
# kB = 1.380649e-3 # units of kg ã² / s²
kB = 1.380649e-27 # units of kg ã² / ps²
tstep = 1 # ps

class Particle():
    def __init__(self, Pot, pos = 0, D = 0.1, T = 300, nsize = 1, dt = tstep):
        """Initializes the simulation

        Arguments:
           Pot (MultiPolynomial):    a polynomial describing a potential function
           pos (np array):           a vector describing the start position
           D (scalar):               diffusion coefficient in amstrongs^2/ps
           T (scalar):               temperature in K
           nsize (scalar):           can magnify the size of noise
           dt (scalar):              timestep in seconds
        """
        self.Pot   = Pot              # potential well
        self.DkT   = D/(kB*T)         # 1/gamma = D/kT
        self.F     = -1*Pot.grad()    # force function
        self.tstep = tstep            # time step
        self.nsize = nsize            # noise size
        self.nsig  = np.sqrt(2*D*dt)  # noise standard deviation

        self.dim   = self.Pot.dim     # dimensionality
        self.pos   = np.array(pos, dtype=np.double) # ND position
        if self.pos.ndim == 0:
            self.pos = self.pos[np.newaxis]

        # initializing the start position to be robust in
        # the case of a too-low-dimensional self.pos
        if (self.pos.shape[0] < self.dim):
            newpos = np.zeros(self.dim, dtype=np.double)
            newpos[:self.pos.shape[0]] = self.pos
            self.pos = newpos
        # self.tot_noise = 0

    def vel(self, pos):
        force = self.F(pos)
        return self.DkT * force

    def noise(self):
        return self.nsize * np.random.normal(0, self.nsig,  self.dim)

    def step(self):
        # # Euler integration
        # force = self.F(self.pos)
        # noise = self.nsize * np.random.normal(0, self.nsig, self.dim)
        # v = self.DkT*force
        # step_size= self.tstep * v + noise

        # Midpoint Integration
        v1 = self.vel(self.pos) # current velocity
        k1 = self.tstep * v1 + self.noise() # Euler prediction of motion
        v2 = self.vel(self.pos + 0.5 * k1) # prediction from midway through the step
        step_size = self.tstep * v2 + self.noise()

        self.pos += step_size

# TriCoeffs = np.array([ [0, 0, 0, 0, 0, 0,   1],   # x^6
#                        [0, 0, 0, 0, 0, 0, 0.2],     # x^5
#                        [0,0,0,0,3, 0.02, -0.46],      # x^4
#                        [0,0,0,0,0.4, 0.12, -1.762],     # x^3
#                        [0,0,3,0.4,0.04, -0.726, -0.1395], # x^2
#                        [0,0,0.2,0.12,5.118, 0.292, 0.405],  # x^1
#                        [1, 0.2, .5, .314, .2565, .114, .7565] # x^0
#                      ])

# input values scaled down  by a factor of 2
# ((x-2)^2+(y)^2)*((x+1)^2+(y+1.73)^2)*((x+2)^2+(y-1.73)^2)
# TriCoeffs = np.array([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.004],
#                       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.002],
#                       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.012,  0.   , -0.007],
#                       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.003,  0.002, -0.055],
#                       [ 0.   ,  0.   ,  0.012,  0.003,  0.001, -0.023, -0.009],
#                       [ 0.   ,  0.   ,  0.002,  0.002,  0.16 ,  0.018,  0.051],
#                       [ 0.004,  0.002,  0.008,  0.01 ,  0.016,  0.014,  0.189]])

# ((x-4)^2+(y)^2)*((x)^2+(y-3)^2)*((x+4)^2+(y)^2)
RipCoeffs = np.array([[    0,     0,     0,     0,     0,     0,     1],
                      [    0,     0,     0,     0,     0,     0,     0],
                      [    0,     0,     0,     0,     3,    -6,   -23],
                      [    0,     0,     0,     0,     0,     0,     0],
                      [    0,     0,     3,   -12,    18,   192,   -32],
                      [    0,     0,     0,     0,     0,     0,     0],
                      [    1,    -6,    41,  -192,   544, -1536,  2304]])

# -2*(x^2+y^2) + 1/8*(x^2+y^2)^2−1/4(x^3−3x*y^2)
T3Coeffs = np.array([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.125],
                     [ 0.   ,  0.   ,  0.   ,  0.   , -0.25 ],
                     [ 0.   ,  0.   ,  0.25 ,  0.   , -2.   ],
                     [ 0.   ,  0.   ,  0.75 ,  0.   ,  0.   ],
                     [ 0.125,  0.   , -2.   ,  0.   ,  0.   ]])

# QCoeffs = np.zeros((5,5))
# QCoeffs[4,4] = -1
# QCoeffs[4,2] = 1
# QCoeffs[2,4] = 1
# QCoeffs[2,2] = -1

# QWell = MultiPolynomial(QCoeffs)
# CromWell next
RWell = 0.001*MultiPolynomial(RipCoeffs)

# if RWell is not None:
#     xs = np.arange(-5, 5, 0.05)
#     Xs2d = np.transpose([np.tile(xs, xs.shape[0]), np.repeat(xs, len(xs))]).reshape((xs.shape[0], xs.shape[0], 2))
#     xxs = Xs2d.reshape(Xs2d.shape[0]* Xs2d.shape[1], Xs2d.shape[2])
#     outs = np.zeros(xxs.shape[0])
#     for i in range(xxs.shape[0]):
#         outs[i] = RWell(xxs[i])
#     outs = outs.reshape(Xs2d.shape[:2])
#     outs = np.clip(outs, 0,5)
#     plt.imshow(outs,cmap='jet')
#     plt.show()


# WWell = MultiPolynomial([            0.5, 0, 0]) # wide
# DWell = 5e-2*MultiPolynomial([5,   0, -1, 0,  0]) # double
# SWell = MultiPolynomial([1,        0, -3, 0,  0]) # shallow?

# OldNWell = MultiPolynomial([5,0, -0.1, 0, 0])
# NewOldNWell = MultiPolynomial([80,0, -0.5, 0, 0]) # narrow (double) well


a = np.array(1e-25, dtype=np.double)
HWell = a*MultiPolynomial([              1, 0, 0]) # harmonic
# NWell = 5e-27*MultiPolynomial([1,  0, -40, 0, 0]) # narrow (double) well
NWell = 1e-27*MultiPolynomial([1,  0, -40, 0, 0]) # not-so-narrow (double) well
TWell = 4e-28*MultiPolynomial([1, 0, -50, 0, 625, 0, 0]) # triple
T3Well = 4e-27*MultiPolynomial(T3Coeffs) # triple
# TWell = 4e-25*MultiPolynomial(TriCoeffs)
RWell = 9e-28*MultiPolynomial(RipCoeffs)

# (x+4)(x+5)*(x-4)(x-5)*x*x has 3 local minima
# --> x^6 - 41 x^4 + 400 x^2


# xs = np.arange(-10, 10, 0.01)
# plt.plot(xs, NWell(xs))
# # plt.plot(xs, TWell(xs))
# # plt.plot(xs, HWell(xs))
# plt.show()


p1 = Particle(NWell, D = 0.01, nsize = 1, pos = 0)
p2 = Particle(HWell, D = 0.1, nsize = 1, pos = -1)

# # Spit out the coordinates (and control the different trajectories...)

npart = 2
particles = [p1,
             p2,
             # p3,
             # ph2,
             # Particle(0.5*SWell, pos = 0.1, nsize = 5, DkT = 0.1),                 # 4
             # Particle(MultiPolynomial([0.5, 1, 4]), pos = -2, nsize = 5e-1),       # 5
             # Particle(MultiPolynomial([0.5, 1, 4]), pos = -2, nsize = 1),          # 6
             # Particle(MultiPolynomial([2, 13, 0]), pos = -0.1, nsize = 1),         # 7
            ][:npart]
nsteps = 552500
# nsteps = 225000
dimensions = 0
for i in range(npart):
    dimensions += particles[i].dim
print("Dimensions:", dimensions)
tracks = np.zeros((nsteps, dimensions), dtype=np.float32)
for i in tqdm(range(tracks.shape[0])):
    j = 0
    # if i == 380:
    #     pdb.set_trace()
    # pdb.set_trace()
    for p in particles:
        p.step()
        pdim = p.dim
        tracks[i,j:j+pdim] = p.pos
        j+= pdim
    # print(("Time %d ps: " % i), tracks[i,:])

def save_file():
    h5file = h5py.File('data/SimOutput.h5', 'w')
    h5file.create_dataset('particle_tracks', data=tracks)
    h5file.close()

# # To open:
# h5file = h5py.File('data/SimOutput.h5', 'r')
# b = h5file['particle_tracks'][:]
# h5file.close()

print("Fraction of time particle 1 spends on the right of 0:", np.sum(tracks[:,0] >= 0)/tracks.shape[0])

plt.plot(np.arange(0,(i+1)*tstep, tstep), tracks)
plt.show()

# Hist, edges = np.histogram(tracks[:,0], bins = 30)
# plt.plot(0.5*(edges[1:]+edges[:-1]),Hist)
# plt.show()

plt.hist2d(tracks[:,0], tracks[:,1], bins = 30)
plt.show()

to_save = input("Do you want to save this particle tracks? (y/n) (If you're in console mode, you can save later manually with 'save_file')  ")
if to_save.lower() == 'y' or to_save.lower() == 'yes':
    save_file()
