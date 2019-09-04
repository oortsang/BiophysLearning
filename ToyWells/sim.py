# Oliver Tsang, Summer 2019
# This file runs a Brownian Dynamics simulation on provided potential wells, some of which are stored in potwells.py.
import numpy as np
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import pdb

from multipoly import MultiPolynomial, PiecewisePolynomial, EasyPiecewise

# Constants
# kB = 1.380649e-23 # units of J/K
# m / s * (a / m) * (s / ps)
#         (1e-10)    (1e12)
# kB = 1.380649e-3 # units of kg ã² / s²
kB = 1.380649e-27 # units of kg ã² / ps²
tstep = 1 # ps

class Particle():
    """Stores the coordinate information as well as physical properties of a particle. Can be an arbitrary number of dimensions."""
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
        self.nsig  = np.sqrt(2*D*dt)  # noise standard deviation from fluctuation-dissipation thm
        self.dim   = self.Pot.dim     # dimensionality
        self.pos   = np.array(pos, dtype=np.double) # n-dim position
        if self.pos.ndim == 0:
            self.pos = self.pos[np.newaxis]

        # initializing the start position to be robust in case self.pos is too few dimensions
        if (self.pos.shape[0] < self.dim):
            newpos = np.zeros(self.dim, dtype=np.double)
            newpos[:self.pos.shape[0]] = self.pos
            self.pos = newpos

    def vel(self, pos):
        """Find the velocity of a particle given the force field"""
        force = self.F(pos)
        return self.DkT * force

    def noise(self):
        """Returns the amount of perturbation caused by noise"""
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

def plot2d(potwell):
    """Plots a 2D potential well"""
    xs = np.arange(-5, 5, 0.05)
    Xs2d = np.transpose([np.tile(xs, xs.shape[0]), np.repeat(xs, len(xs))])\
             .reshape((xs.shape[0], xs.shape[0], 2)) # make a grid
    xxs = Xs2d.reshape(Xs2d.shape[0]* Xs2d.shape[1], Xs2d.shape[2])
    outs = np.zeros(xxs.shape[0])
    for i in range(xxs.shape[0]):
        outs[i] = potwell(xxs[i])
    outs = outs.reshape(Xs2d.shape[:2])
    outs = np.clip(outs, 0,5)
    plt.imshow(outs,cmap='jet')
    plt.show()


# ((x-4)^2+(y)^2)*((x)^2+(y-3)^2)*((x+4)^2+(y)^2)
from potwells import RipCoeffs
RWell = 0.001*MultiPolynomial(RipCoeffs)

# Piecewise potential landscape with 3 wells in 2D...
from potwells import PWell

# Piecewise potential landscape with 3 wells in 2D...
from potwells import p_trip

# Narrow-ish double well
from potwells import NWell

# Harmonic well
from potwells import HWell

# Hexagonally distributed potential wells...
def polyg_well_gen(n_points = 6, cr = 1):
    """Input: n_points - number of points to put on the circumference of a circle
    cr - circumradius of the polygon / radius of the circle where the points lie
    """
    centers = cr*np.array([(np.cos(2*np.pi*k/n_points), np.sin(2*np.pi*k/n_points)) for k in range(n_points)])
    barriers = []
    for k in range(n_points):
        # (c,s) . (x, y) = 0 are the dividing lines
        # (-sd, cd) . (x, y) is the projection onto the normal of the dividing line
        angle = (2*np.pi * (k+1/2)  / n_points) % (2*np.pi)
        s = np.sin(angle)
        c = np.cos(angle)
        d = np.sign(c) # direction
        tmp_barr = MultiPolynomial([[0,-s*d],[c*d,0]]) # [[xy, x], [y, 1]]
        barriers.append(tmp_barr)





# plot a couple of the wells
xs = np.arange(-10, 10, 0.01)
plt.plot(xs, NWell(xs))
plt.plot(xs, PWell(xs))
# plt.plot(xs, TWell(xs))
plt.show()

########## Prepare and run the simulation ####################################

# Set up the particles
p1 = Particle(NWell, D = 0.01, nsize = 1, pos = 0)
p2 = Particle(HWell, D = 0.1, nsize = 1, pos = -1)

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
# nsteps = 352500
nsteps = 135000
dimensions = 0
for i in range(npart):
    dimensions += particles[i].dim
print("Dimensions:", dimensions)
tracks = np.zeros((nsteps, dimensions), dtype=np.float32)

# run the actual simulation
for i in tqdm(range(tracks.shape[0])):
    j = 0
    for p in particles:
        p.step()
        pdim = p.dim
        tracks[i,j:j+pdim] = p.pos
        j+= pdim

        
########## Visualization and saving ##########################################

def save_file():
    h5file = h5py.File('data/SimOutput.h5', 'w')
    h5file.create_dataset('particle_tracks', data=tracks)
    h5file.close()

# # To open:
# h5file = h5py.File('data/SimOutput.h5', 'r')
# b = h5file['particle_tracks'][:]
# h5file.close()

print("Fraction of time particle 1 spends on the right of 0:", (tracks[:,0] >= 0).mean())

plt.plot(np.arange(0,(i+1)*tstep, tstep), tracks)
plt.show()

plt.hist2d(tracks[:,0], tracks[:,1], bins = 60, cmap = 'jet')
plt.show()

to_save = input("Do you want to save this particle tracks? (y/n) (If you're in "
                "console mode, you can save later manually with 'save_file')  ")
if to_save.lower() == 'y' or to_save.lower() == 'yes':
    save_file()
