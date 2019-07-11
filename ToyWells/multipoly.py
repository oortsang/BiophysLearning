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
        orig_dim  = x.ndim
        x = np.atleast_2d(x)
        input_dim = x.shape[-1]
        # need to take care of the dimension business...
        res = []
        # pdb.set_trace()
        for i, xi in enumerate(x): # for each example
            ind_x_pows = np.vander(xi, self.degree)
            x_pows = ind_x_pows[0]
            for i in range(1, xi.shape[0]):
                x_pows = x_pows[...,np.newaxis] * ind_x_pows[i]
            tmp = self.coeffs * x_pows
            res.append(tmp)
        res = np.array(res)
        if res.ndim < 4:
            res = res[np.newaxis,:]
        #or orig_dim > coeffs.ndim:
        res = res.squeeze()
        leave_dims = 1 if self.return_vector else 0
        dimgap = self.coeffs.ndim - input_dim + (orig_dim-1)
        leave_dims += dimgap
        pdb.set_trace()

        # summed = res.sum(axis=tuple(range(res.ndim-1, 1,-1)))
        leave_dims = np.max((0, leave_dims))
        summed = res.sum(axis=tuple(range(res.ndim-1, leave_dims-1, -1)))
        # add a correction to distinguish Vector-Valued 2D from scalar 3D...
        return np.squeeze(summed)

    def grad(self):
        pass
    
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


F = MultiPolynomial([[0,1,2],[1,2,0],[1,3,4]])
H = MultiPolynomial(np.array([[[1,0],[0,1]],[[0,1],[1,0]]]))
J = MultiPolynomial([1,0,0])
print(F([2,3])) # expect 72
print(H([1,1,1])) # expect 4
print(H([[1,1,1],[0,0,0]])) # expect [4 0]
print(J(3)) # expect 9
print(J([3])) # expect 9
print(J([3,4])) # expect [9 16]
