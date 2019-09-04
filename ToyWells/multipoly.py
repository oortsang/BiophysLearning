import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Polynomials in multiple variables
# for clean manipulation of potential wells
class MultiPolynomial():
    # Initialize with an N-D array of coefficients
    # if "return_vector" is set to True, the evaluation will not sum over different variables
    def __init__(self, coeffs, return_vector = False):
        self.coeffs = np.array(coeffs)
        self.return_vector = return_vector
        self.degree = np.max(self.coeffs.shape) # informally going to ask that each axis be equally long...
        if self.coeffs.ndim >= 1:
            self.dim = self.coeffs.ndim
        else:
            self.dim = 1

        self.return_dim = self.coeffs.shape[0] if return_vector else 1

    def eval(self, x):
        x = np.array(x)
        x = np.atleast_1d(x)
        # pdb.set_trace()
        if self.coeffs.ndim == 1 and x.ndim == 1 and x.shape[0] > 1:
            x = x[:,np.newaxis]
        orig_dim  = x.ndim
        x = np.atleast_2d(x)
        input_dim = x.shape[-1]

        res = []
        coeffs = self.coeffs
        if not self.return_vector:
            coeffs = self.coeffs[np.newaxis,:]
        # iterate through different coefficient stuff -- for vector-valued functions
        # newaxis above is for scalar functions to maintain behavior without the for loop
        # for c in coeffs:
        #     for i, xi in enumerate(x): # for each example
        for i, xi in enumerate(x): # for each example
            ind_x_pows = np.vander(xi, self.degree)
            x_pows = ind_x_pows[0]
            for i in range(1, xi.shape[0]):
                x_pows = x_pows[...,np.newaxis] * ind_x_pows[i]

            for c in coeffs:
                tmp = c * x_pows
                res.append(tmp)
        res = np.array(res)
        if res.ndim < 4:
            res = res[np.newaxis,:]
        res = res.squeeze()
        # leave_dims = np.max((self.return_vector, self.coeffs.ndim - input_dim + (orig_dim-1)))
        leave_dims = np.max((0, self.coeffs.ndim - input_dim + (orig_dim-1)))
        summed = res.sum(axis=tuple(range(res.ndim-1, leave_dims-1, -1)))
        # import pdb; pdb.set_trace()
        if self.return_vector:
            summed = summed.reshape((summed.shape[0]//input_dim, input_dim))
        # add a correction to distinguish Vector-Valued 2D from scalar 3D...
        return np.squeeze(summed)

    def grad(self):
        # pdb.set_trace()
        if self.return_vector:
            print("Jacobians are too hard at the moment")
            return None
        vdim = self.coeffs.ndim
        new_coeffs = np.zeros((vdim, *self.coeffs.shape))
        dlen = self.coeffs.shape[-1]
        diff_mat = np.zeros((dlen, dlen))
        idcs = np.arange(dlen-1)
        diff_mat[idcs+1, idcs] = np.arange(dlen-1, 0, -1)

        for dim in range(vdim):
            tmp_coeffs = self.coeffs.swapaxes(0,dim)
            new_coeffs[dim] = np.dot(diff_mat, tmp_coeffs).swapaxes(0,dim)
        if vdim == 1:
            new_coeffs = new_coeffs[0]
        return MultiPolynomial(new_coeffs, return_vector = vdim > 1)

    def __call__(self, x):
        # convenient syntax like f(3, 4)
        return self.eval(x)

    def __rmul__(self, obj):
        if obj.__class__ != self.__class__:
            newcoeffs = obj * self.coeffs
            return MultiPolynomial(np.array(newcoeffs))

    def __mul__(self, obj):
        if obj.__class__ != self.__class__:
            newcoeffs = obj * self.coeffs
            return MultiPolynomial(np.array(newcoeffs))

    def __repr__(self):
        ret_string = "MultiPolynomial object with %d variables; " % (self.dim)
        ret_string += "returns a " + ("vector" if self.return_vector else "scalar") + "\n"
        return ret_string


class PiecewisePolynomial():
    """A holder for multidimensional polynomials that holds two polynomials (or Piecewise Polynomials): p1 for when c>=0 and p2 when c < 0"""
    def __init__(self, p1, p2, c):
        self.p1 = p1
        self.c = c
        self.p2 = p2
        if p1.dim != p2.dim:
            print("Uh oh, the two polynomials you gave me have different dimensions...")
        self.dim = p1.dim
        self.return_dim = self.p1.return_dim

    def eval(self, x):
        """Returns the value of the function at the given point. The function is decided by the value of the c function (which could be a piecewise polynomial itself)"""
        xx = np.array(x)
        if xx.ndim > 1 or (xx.ndim == 1 and self.p1.dim == 1): # If the inputs xx come as a list,
            val_shape = (*xx.shape[:-1], self.p1.return_dim)   #  prepare the shape of the output
            if xx.ndim == 1 or self.p1.return_dim == 1:        # If our input is low-dimensional,
                val_shape = (x.shape[0], )                     #  shape the outputs appropriately
            val = np.zeros(val_shape, dtype = np.double)
            p1_idcs = self.c(x) >= 0                           # Decide which points get which fn
            p2_idcs = np.logical_not(p1_idcs)
            if x.ndim == 1 or self.return_dim == 1: # massage
                p1_idcs = p1_idcs.flatten()
                p2_idcs = p2_idcs.flatten()

            if np.any(p1_idcs):
                val[p1_idcs] = self.p1(x[p1_idcs])
            if np.any(p2_idcs):
                val[p2_idcs] = self.p2(x[p2_idcs])
        else:
            val = (self.p1 if self.c(xx) >= 0 else self.p2) (xx)
        return val

    def grad(self):
        """Return gradient as a piecewise function. Doesn't check for continuity."""
        g1 = self.p1.grad()
        g2 = self.p2.grad()
        return self.__class__(g1, g2, self.c)

    def __call__(self, x):
        return self.eval(x)

    def __mul__(self, a):
        return self.__class__(a*self.p1, a*self.p2, self.c)

    def __rmul__(self, a):
        return self.__class__(a*self.p1, a*self.p2, self.c)

    def __repr__(self):
        return "Piecewise Polynomial"

class EasyPiecewise():
    """Acts as a Piecewise Polynomial but uses a different parameterization."""
    def __init__(self, ps, cs):
        """Input:
                 ps - array with m polynomials [p0, p1, ...,  p(m-1)]
                 cs - a (vectorized) function that takes the input point
                      and returns and integer in [0, m-1]
        """
        if (len(ps) <=1):
            print("Expected a list of polynomials for ps, but the dimensions aren't right...")
            return
        self.ps = ps
        self.cs = cs
        self.m = len(ps)
        self.dim = ps[0].dim
        self.return_dim = ps[0].return_dim

    def eval(self, x):
        """Returns the value of the function at the given point. The function is decided by the value of the c function (which could be a piecewise polynomial itself)"""
        xx = np.array(x) # so that we can check the ndim even if the input isn't a numpy array
        if xx.ndim > 1 or (xx.ndim == 1 and self.dim == 1): # If the inputs xx come as a list,
            val_shape = (*xx.shape[:-1], self.return_dim)   #  prepare the shape of the output
            if x.ndim == 1 or self.return_dim == 1:         # If our input is low-dimensional,
                val_shape = (x.shape[0], )                  #  shape the outputs appropriately
            val = np.zeros(val_shape, dtype = np.double)

            conds = self.cs(x)
            for pi in range(len(self.ps)):
                pi_idcs = (conds == pi)
                if x.ndim == 1 or self.return_dim == 1: # massage
                    pi_idcs = pi_idcs.flatten()
                if np.any(pi_idcs):
                    val[pi_idcs] = self.ps[pi](x[pi_idcs])
        else:
            idx = self.cs(xx)
            val = (self.ps[idx]) (xx)
        return val

    def grad(self):
        """Return the gradient"""
        gs = [p.grad() for p in self.ps]
        return self.__class__(gs, self.cs)

    def __call__(self, x):
        return self.eval(x)

    def __mul__(self, a):
        return self.__class__([a*p for p in self.ps], self.c)

    def __rmul__(self, a):
        return self.__class__([a*p for p in self.ps], self.c)




# p1 = MultiPolynomial([1, -4, 4])
# p2 = MultiPolynomial([1,  4, 4])
# c  = MultiPolynomial([1,0])
# pp = PiecewisePolynomial(p1, p2, c)
# xs = np.arange(-4,4,0.1)
# ys = pp(xs)
# plt.plot(xs, ys)
# plt.show()

# w1 = MultiPolynomial([[0, 0, 1], [0, 0, -14], [1, 8, 0]])
# w2 = MultiPolynomial([[0, 0, 1], [0, 0,  14], [1, 8, 0]])
# w3 = MultiPolynomial([[0, 0, 0.75], [0, 0, 0], [0.75, -6, -72]])
# c1  = MultiPolynomial([[0,1],[0,0]])

# # c2  = MultiPolynomial([[0,0],[1,-2]])
# c21 = MultiPolynomial([[0,-1],[2,5]])
# c22 = MultiPolynomial([[0,1],[2, 5]])
# c2c = MultiPolynomial([[0,1],[0,0]])
# c2 = PiecewisePolynomial(c21, c22, c2c)

# p1  = PiecewisePolynomial(w1, w2, c1)
# p   = PiecewisePolynomial(w3, p1, c2)

# xs1d = np.arange(-12, 12, 0.5)
# xs = np.transpose([np.tile(xs1d, xs1d.shape[0]), np.repeat(xs1d, xs1d.shape[0])]).reshape((xs1d.shape[0], xs1d.shape[0], 2))
# zs = p(xs.reshape(xs.shape[0]*xs.shape[1], xs.shape[2])).reshape(xs.shape[:-1])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xs[:,:,0], xs[:,:,1], zs, cmap = 'jet')
# # ax.plot_wireframe(xs[:,:,0], xs[:,:,1], zs)
# plt.show()

# F = MultiPolynomial([[0,1,2],[1,2,0],[1,3,4]]) # 2D
# H = MultiPolynomial(np.array([[[1,0],[0,1]],[[0,1],[1,0]]])) # 3D but with relatively low degree
# J = MultiPolynomial([1,2,1]) # 1D Polynomial

# print(H([-1,4,2])) # expect scalar
# G = F.grad()
# G0 = MultiPolynomial(G.coeffs[0])
# G1 = MultiPolynomial(G.coeffs[1])
# print(G([-1, 4])) # expect 2D vector


# # G = MultiPolynomial([[[0,1],[1,1]], [[1,0],[0,1]]], return_vector = True)
# # G0 = MultiPolynomial([[0,1],[1,1]])
# # G1 = MultiPolynomial([[1,0],[0,1]])

# print(G([1,2]))  # should give [4 3]
# print(G0([1,2])) # should get 4
# print(G1([1,2])) # should get 3

# # scalar outputs working nicely
# print("Want 72       - ", F([2,3]))             # # 1 2D input  -> 1 1D output
# print("Want [72 296] - ", F([[2,3],[4,5]]))     # # 2 2D inputs -> 2 1D outputs
# print("Want 4        - ", H([1,1,1]))           # # 1 3D input  -> 1 1D output
# print("Want [4 0]    - ", H([[1,1,1],[0,0,0]])) # # 2 3D inputs -> 2 1D outputs
# print("Want 16       - ", J(3))                 # # 1 1/0D input-> 1 1D output
# print("Want 16       - ", J([3]))               # # 1 1D input  -> 1 1D output
# print("Want [16 25]  - ", J([[3],[4]]))         # # 2 1D inputs -> 2 1D outputs
# print("Want [16 25]  - ", J([3,4]))             # # 2 1D inputs -> 2 1D outputs
