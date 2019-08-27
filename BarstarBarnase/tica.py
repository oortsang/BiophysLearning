import numpy as np
import h5py as h5
from scipy.linalg import eig, eigh, inv

import matplotlib.pyplot as plt

class TICA():
    """Implemented based on description in Frank Noé, Machine Learning for Molecular Dynamics on Long Timescales, 2018 (https://arxiv.org/abs/1812.07669)"""
    def __init__(self, x, dt, kinetic_map_scaling = True):
        """Takes time series data"""
        # coo = np.cov(x[:-dt, :].T)
        # cot = np.cov(x[dt:, :].T)
        xo = x[:-dt]
        xt = x[dt:]
        muo = np.mean(xo, axis = 0)
        mut = np.mean(xt, axis = 0)
        coo = (xo-muo).T @ (xo-muo) + (xt-mut).T @ (xt-mut) # 0.05*np.eye(muo.shape[0])
        cot = (xo-muo).T @ (xt-mut) + (xt-mut).T @ (xo-muo)

        eigvals, eigvecs = eig(cot, coo)
        # eigvals, eigvecs = eigh(cot, coo, eigvals_only = False) # disagrees with eig...


        self.kinetic_map_scaling = kinetic_map_scaling

        # import pdb; pdb.set_trace()
        ordering = np.argsort(-(eigvals)) # descending order
        self.eigvals = np.real(eigvals[ordering])
        self.eigvecs = np.real(eigvecs[ordering]) # eigvecs are [:,i] so we reorder the (:) dim
        print(self.eigvals)

        self.evec_inv = inv(self.eigvecs)
        # find the propagator
        self.prop = inv(coo) @ cot # see paper...

    def transform(self, xx):
        """Transforms to tIC space"""
        xp =  xx @ self.evec_inv.T
        if self.kinetic_map_scaling:
            xp = xp * self.eigvals[np.newaxis]
        return xp

    def inv_transform(self, xx):
        """Transforms from tIC space to the input space"""
        if self.kinetic_map_scaling:
            xx = xx / self.eigvals[np.newaxis]
        xp = xx @ self.eigvecs.T
        return xp


if __name__ == "__main__":
    fname = "data/ConvolvedSimFeatures.h5"
    # fname = "data/NormalizedSimFeatures.h5"
    # fname = "data/SimOutput_3well500k.h5"
    data = h5.File(fname, 'r')["particle_tracks"]
    dt = 50

    tica = TICA(data, dt)
    xp = tica.transform(data)
    H, xe, ye = np.histogram2d(xp[:,0], xp[:,1], bins = 50)
    plt.imshow(H, cmap = 'jet')
    plt.show()
