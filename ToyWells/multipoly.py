import pdb
import numpy as np
import matplotlib.pyplot as plt

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
        for c in coeffs:
            for i, xi in enumerate(x): # for each example
                ind_x_pows = np.vander(xi, self.degree)
                x_pows = ind_x_pows[0]
                for i in range(1, xi.shape[0]):
                    x_pows = x_pows[...,np.newaxis] * ind_x_pows[i]
                tmp = c * x_pows
                res.append(tmp)
        res = np.array(res)
        if res.ndim < 4:
            res = res[np.newaxis,:]
        res = res.squeeze()
        leave_dims = np.max((0, self.coeffs.ndim - input_dim + (orig_dim-1)))
        summed = res.sum(axis=tuple(range(res.ndim-1, leave_dims-1, -1)))
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

    def __rmul__(self, scalar):
        newcoeffs = scalar * self.coeffs
        return MultiPolynomial(np.array(newcoeffs))

    def __mul__(self, scalar):
        newcoeffs = scalar * self.coeffs
        return MultiPolynomial(np.array(newcoeffs))

    def __repr__(self):
        ret_string = "MultiPolynomial object with %d variables; " % (self.dim)
        ret_string += "returns a " + ("vector" if self.return_vector else "scalar") + "\n"

        return ret_string


F = MultiPolynomial([[0,1,2],[1,2,0],[1,3,4]]) # 2D
H = MultiPolynomial(np.array([[[1,0],[0,1]],[[0,1],[1,0]]])) # 3D but with relatively low degree
J = MultiPolynomial([1,2,1]) # 1D Polynomial

print(H([-1,4,2])) # expect scalar
G = F.grad()
G0 = MultiPolynomial(G.coeffs[0])
G1 = MultiPolynomial(G.coeffs[1])
print(G([-1, 4])) # expect 2D vector


# G = MultiPolynomial([[[0,1],[1,1]], [[1,0],[0,1]]], return_vector = True)
# G0 = MultiPolynomial([[0,1],[1,1]])
# G1 = MultiPolynomial([[1,0],[0,1]])

print(G([1,2]))  # should give [4 3]
print(G0([1,2])) # should get 4
print(G1([1,2])) # should get 3

# scalar outputs working nicely
print("Want 72       - ", F([2,3]))             # # 1 2D input  -> 1 1D output
print("Want [72 296] - ", F([[2,3],[4,5]]))     # # 2 2D inputs -> 2 1D outputs
print("Want 4        - ", H([1,1,1]))           # # 1 3D input  -> 1 1D output
print("Want [4 0]    - ", H([[1,1,1],[0,0,0]])) # # 2 3D inputs -> 2 1D outputs
print("Want 16       - ", J(3))                 # # 1 1/0D input-> 1 1D output
print("Want 16       - ", J([3]))               # # 1 1D input  -> 1 1D output
print("Want [16 25]  - ", J([[3],[4]]))         # # 2 1D inputs -> 2 1D outputs
print("Want [16 25]  - ", J([3,4]))             # # 2 1D inputs -> 2 1D outputs
