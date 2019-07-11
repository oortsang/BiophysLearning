import numpy as np
import matplotlib.pyplot as plt

# Polynomials in multiple variables
# for clean manipulation of potential wells
class MultiPolynomial():
    # Initialize with a 2D array of coefficients
    # if "return_vector" is set to True, the evaluation will not sum over different variables
    def __init__(self, coeffs, return_vector = False):
        self.coeffs = np.array(coeffs)
        self.set_degree()
        self.return_vector = return_vector
        if self.coeffs.ndim > 1:
            self.dim = self.coeffs.shape[0]
        else:
            self.dim = 1
            self.coeffs = self.coeffs[np.newaxis,:] # make it 2D

    def set_degree(self):
        if self.coeffs.shape[-1] == 1 and self.coeffs[0] == 0:
            self.degree = -np.inf
        else: # assume no leading zeros
            self.degree = self.coeffs.shape[-1] - 1

    # return a new polynomial that is the derivative of the current polynomial
    def grad(self, return_vector = True):
        newcoeffs = np.array(self.coeffs[:,:-1])
        exponents = np.arange(self.degree, 0, -1)
        newcoeffs = np.multiply(newcoeffs, exponents)
        # for i in range(self.dim):
        #     dimcoeffs = np.zeros(self.coeffs.shape[-1]-1)
        #     exponents = np.arange(self.degree, 0, -1)
        #     dimcoeffs = exponents * self.coeffs[i, :-1]
        #     newcoeffs[i] = dimcoeffs
        return MultiPolynomial(newcoeffs, return_vector = return_vector)

    # evaluate the polynomial at x (note that this broadcasts cleanly)
    def eval(self, x):
        x = np.array(x)
        tmp = np.zeros(self.dim)
        for i in range(self.degree, 0-1, -1): # degree inclusive -> 0 inclusive (-1 exclusive)
            tmp = np.multiply(tmp, x) + self.coeffs[:,self.degree - i]
            # print(tmp)
        if self.dim > 1 and not self.return_vector:
            return tmp.sum(-1)
        else:
            return tmp
        

    def __call__(self, x):
        # convenient syntax like f(3.4)
        return self.eval(x)

    def __rmul__(self, scalar):
        newcoeffs = scalar * self.coeffs
        return MultiPolynomial(np.array(newcoeffs))

    def __mul__(self, scalar):
        newcoeffs = scalar * self.coeffs
        return MultiPolynomial(np.array(newcoeffs))

    def __repr__(self):
        ret_string = "MultiPolynomial object with %d variables of degree %d; " % (self.dim, self.degree)
        ret_string += "returns a " + ("vector" if self.return_vector else "scalar") + "\n"
        for i in range(self.dim):
            line = "  "
            var = "(x_%d)" % (i+1)
            cs = self.coeffs[i]
            for j in range(cs.shape[0]):
                e = self.degree - j
                c = cs[j]
                if c == 0:
                    continue
                else:
                    if j != 0 and c > 0:
                        line += " + "
                    elif j != 0 and c < 0:
                        line += " - "
                    c = np.abs(c)
                    if e == 0:
                        line += "%.2f" % c
                    elif e == 1:
                        line += "%.2f %s" % (c, var)
                    else:
                        line += "%.2f %s^%d" % (c, var, e)
            ret_string += line + "\n"
        return ret_string
