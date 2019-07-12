import numpy as np
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

# Potential functions

# Univariate polynomials
# for clean manipulation of potential wells
# could maybe later extend to descirbe multiple dimensions...
class Polynomial():
    # Initialize with an array of coefficients
    def __init__(self, coeffs):
        self.coeffs = coeffs
        if len(coeffs) == 1 and coeffs[0] == 0:
            self.degree = -np.inf
        else: # assume no leading zeros
            self.degree = len(coeffs)-1

    # return a new polynomial that is the derivative of the current polynomial
    def deriv(self):
        newcoeffs = []
        for i in range(self.degree, 0, -1):
            newcoeffs.append(i * self.coeffs[self.degree - i])
        return Polynomial(np.array(newcoeffs))

    # evaluate the polynomial at x (note that this broadcasts cleanly)
    def eval(self, x):
        res = 0
        for i in range(self.degree, 0-1, -1): # degree inclusive -> 0 inclusive (-1 exclusive)
            res = res * x + self.coeffs[self.degree - i]
        return res

    def __call__(self, x):
        # convenient syntax like f(3.4)
        return self.eval(x)

    def __rmul__(self, scalar):
        newcoeffs = []
        for i in range(self.degree+1):
            newcoeffs.append(scalar * self.coeffs[i])
        return Polynomial(np.array(newcoeffs))

# f = Polynomial(np.array([1,4,4]))
# g = f.deriv()
# xs = np.arange(-10,10, 0.1)

# Brownian motion / taking random steps

kB = 1.380649e-23 # units of J/K
class Particle():
    def __init__(self, Pot, pos = 0, v = 0, DkT = 1, nsize = 1e-2):
        self.Pot = Pot        # potential well
        self.F = -1*Pot.deriv() # force function
        self.pos = pos        # 1D position
        self.v = v            # velocity - um/ns ~ m/ms
        self.DkT = DkT        # 1/gamma = D/kT
        self.nsize = nsize    # noise size

    def step(self, dt): # try dt in s
        # dx/dt = D/kT F(x) + r(t)
        force = self.F(self.pos)
        noise = self.nsize * np.random.normal(0,1)
        self.v = self.DkT*force + noise
        self.pos += dt * self.v

HWell = Polynomial([         1, 0, 0])
DWell = 5e-2*Polynomial([5,   0, -1, 0, 0])
SWell = Polynomial([1,   0, -3, 0, 0])

p1 = Particle(Polynomial([0.5, 1, 4]), pos = -2, nsize = 1e-1)
p2 = Particle(SWell, nsize = 5, DkT = 0.1)
p3 = Particle(2*DWell, nsize = 0.5, DkT = 0.1)
p4 = Particle(0.5*SWell, nsize = 3, DkT = 0.1, pos = -2)

xs = np.arange(-3, 3, 0.05)
plt.plot(xs, HWell(xs))
plt.plot(xs, DWell(xs))
plt.plot(xs, SWell(xs))
plt.show()

# Spit out the coordinates (and control the different trajectories...)

npart = 15
particles = [p1, p2, p3, p4,
             Particle(Polynomial([0.01, 0, -1]), pos = 0, nsize = 1),     # 5 # moves around a lot
             Particle(Polynomial([0.5, 1, 4]), pos = -2, nsize = 1),      # 6
             Particle(Polynomial([2, 13, 0]), pos = -0.1, nsize = 1),     # 7
             Particle(Polynomial([0.5, 1, 4]), pos = 0, nsize = 1e-1),    # 8
             Particle(Polynomial([0.5, 1, -1]), pos = 0.2, nsize = 1e-1), # 9
             Particle(Polynomial([0.5, 1, 1]), pos = -0.3, nsize = 1e-1), # 10
             Particle(Polynomial([0.7, 1, 1]), pos = 0, nsize = 1e-1, DkT = 0.0001), # 11
             Particle(Polynomial([0.5, 1, 1]), pos = 0, nsize = 1, DkT = 10), # 12
             Particle(Polynomial([3, 1, 1]), pos = 0, nsize = 1),      # 13
             Particle(Polynomial([10, -10, -1]), pos = 0, nsize = 1),  # 14
             Particle(Polynomial([0.5, 1, 1]), pos = -0.1, nsize = 1), # 15
             # Particle(Polynomial([0.5, 1, -1]), pos = 0.5, nsize = 1)  # 16
            ][:npart]
tstep = 1e-2
nsteps = 100000
tracks = np.zeros((nsteps, npart), dtype=np.float32)
for i in tqdm(range(tracks.shape[0])):
    for j, p in enumerate(particles):
        p.step(tstep)
        tracks[i,j] = p.pos

h5file = h5py.File('SimOutput.h5', 'w')
h5file.create_dataset('particle_tracks', data=tracks)
h5file.close()

# # To open:
# h5file = h5py.File('SimOutput.h5', 'r')
# b = h5file['particle_tracks'][:]
# h5file.close()


plt.plot(tracks)
plt.show()

