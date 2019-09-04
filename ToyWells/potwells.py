# Place to put all the junk from sim.py
import numpy as np
from multipoly import MultiPolynomial, PiecewisePolynomial, EasyPiecewise

##############################################################################

########## Defining some coefficients ########################################

# (x+4)(x+5)*(x-4)(x-5)*x*x has 3 local minima
# --> x^6 - 41 x^4 + 400 x^2
TWell = 4e-28 * MultiPolynomial([1, 0, -50, 0, 625, 0, 0]) # triple



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


########## Defining regular potential wells based on coefficients ############

a = np.array(1e-25, dtype=np.double)
HWell  = a * MultiPolynomial([              1, 0, 0]) # harmonic
NWell  = 1e-27 * MultiPolynomial([1,  0, -40, 0, 0]) # not-so-narrow (double) well
T3Well = 4e-27 * MultiPolynomial(T3Coeffs) # triple
RWell  = 9e-28 * MultiPolynomial(RipCoeffs)


########## Piecewise Functions ###############################################

# Piecewise potential landscape with 3 wells in 2D...
height = 8e-26
lwell  = height * MultiPolynomial([1.5, 16, 32])
mwell  = height * MultiPolynomial([1, 0,  -11.5])
rwell  = height * MultiPolynomial([1.5,-16, 32])
c1     = MultiPolynomial([-1,3])
c2     = MultiPolynomial([-1,-3])
PLWell = PiecewisePolynomial(mwell,  rwell, c1)
PWell  = PiecewisePolynomial(lwell, PLWell, c2)

# Piecewise potential landscape with three wells in 2D
scale  = 2e-26
w1     = scale * MultiPolynomial([[0, 0, 1], [0, 0, -14], [1, 8, 0]])
w2     = scale * MultiPolynomial([[0, 0, 1], [0, 0,  14], [1, 8, 0]])
w3     = scale * 0.75 * MultiPolynomial([[0, 0, 0.75], [0, 0, 0], [0.75, -6, -72]])
c1     = MultiPolynomial([[0,1],[0,0]])
c21    = MultiPolynomial([[0,-1],[2,5]])
c22    = MultiPolynomial([[0,1],[2, 5]])
c2c    = MultiPolynomial([[0,1],[0,0]])
c2     = PiecewisePolynomial(c21, c22, c2c)
p1     = PiecewisePolynomial(w1, w2, c1)
p_trip = PiecewisePolynomial(w3, p1, c2)

# Generate regular-polygonally-distributed potential wells
def polyg_well_gen(n_points = 6, cr = 1):
    """Generates regular-polygonally-distributed potential wells
    Input: n_points (int) - number of points to put on the circumference of a circle
           cr (float) - circumradius of the polygon / radius of the circle where the points lie
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
        tmp_barr = [[0,-s*d],[c*d,0]] # [[xy, x], [y, 1]]
        barriers.append(tmp_barr)
    selector = MultiPolynomial(barriers, return_vector = True)
    return selector




########## Commented-out potential wells that could be useful ################

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
