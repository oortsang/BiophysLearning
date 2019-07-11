import os
import numpy as np
import h5py as h5
from torch.utils import data
from sklearn.decomposition import PCA

class MyData(data.TensorDataset):
    def __init__(self, fname, transform = None):
        self.archive = h5.File(fname, 'r')
        self.data = self.archive['particle_tracks']
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum

    def __len__(self):
        return len(self.data)

    def close(self):
        self.archive.close()

def normalize(dataset):
    normdata = np.zeros(dataset.data.shape, dtype = np.float32)
    for i in range(dataset.data.shape[1]):
        mu  = (dataset[:,i]).mean()
        sig = (dataset[:,i]).std()
        normdata[:,i] = (dataset[:,i]-mu)/sig
    return normdata[500:] # smooth out the beginning

def pcaify(dataset):
    pca = PCA()
    pca_latent = pca.fit_transform(dataset.data[:])
    # pca_latent[:,n_z:] = 0
    # pca_guesses = pca.inverse_transform(pca_latent)
    return pca_latent[500:]



rawfile = "SimOutput.h5"
norfile = "NormalizedSimOutput.h5"
pcafile = "PCANormdSimOutput.h5"

if os.path.getmtime(rawfile) > os.path.getmtime(norfile):
    # if the raw file is more recent than the normalized file, redo it
    print("Normalizing the simulation data...")
    raw_sim_data = MyData(rawfile)
    normalized_data = normalize(raw_sim_data)

    h5file = h5.File(norfile, 'w')
    h5file.create_dataset('particle_tracks', data=normalized_data)
    h5file.close()

if os.path.getmtime(rawfile) > os.path.getmtime(pcafile):
    # if the pca file is more recent than the normalized file, redo it
    print("Normalizing the simulation data...")
    raw_sim_data = MyData(rawfile)
    pca_data = pcaify(raw_sim_data)

    h5file = h5.File(pcafile, 'w')
    h5file.create_dataset('particle_tracks', data=pca_data)
    h5file.close()
    

raw_sim_data = MyData(rawfile)
sim_data = MyData(norfile)
pca_sim_data = MyData(pcafile)


# # open the simulation output
# h5file = h5py.File('SimOutput.h5', 'r')
# data = h5file['particle_tracks'][:]
# h5file.close()
