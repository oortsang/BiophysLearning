# Oliver Tsang, Summer 2019
# (Using Python 3)
# Mostly copied from Ziwei He's make_msm.py file (which is in Python 2)

import mdtraj as md
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm # for a nice progress bar
import h5py

# File names
top_file  = "go_bnbs_complex.pdb"
traj_file = "MD/1brs_cat_%d.dcd" % 0
cps_file = "My Contact Pairs/cpall.npy"

# Initialize files
top = md.load_pdb(top_file)
traj = md.load_dcd(traj_file, top = top)

def get_contact_pairs():
    """
    Return pairs of AAs that are within the cutoff distance for a duration lasting longer
    than the cutoff time (10 frames = 10 ns). This is meant to select narrow down the num-
    ber of features we deal with later.
    """
    cutoff_dist = 1.2 # nm
    cutoff_time = 10  # timesteps = ns
    pairs = []
    pad = lambda nparr, length: np.concatenate((np.array([-1]), nparr, np.array([length])), 0)
    for ni in tqdm(range(0, 110)):
        for si in range(110, 199):
            ds = md.compute_distances(traj, ((ni,si),)).flatten()
            # Keep track of time spent outside the cutoff.. reasons become clear shortly
            outside_cutoff = ds > cutoff_dist
            out_idcs = np.where(outside_cutoff)[0]

            # find the durations between sejours outside the cutoff
            # (i.e., how many timesteps are spent within the cutoff)
            # Padding creates virtual points at -1 and the end - we say
            # these points are outside the cutoff, just to be sure.
            # Subtract 1 because we expect a difference of one in a contiguous section
            durs = np.diff(pad(out_idcs, ds.shape[0])) - 1

            if durs.max() > 10:
                pairs.append((ni, si))
    print("Found %d contact pairs out of a possible %d" % (len(pairs), 110*89))
    np.save(cps_file, pairs)
    return pairs

def load_contact_pairs(file=cps_file):
    cps = np.load(cps_file)
    return cps

# def group_pairs():
#     """Return the unique contact pairs since get_contact_pairs gives duplicates"""
#     pass
def create_array(contact_pairs, traj = traj):
    dist_array = np.zeros((len(contact_pairs), traj.n_frames), dtype = np.float32)
    for i, p in enumerate(contact_pairs):
        dist_array[i] = md.compute_distances(traj, (p,)).flatten()
    dist_array = dist_array.T
    h5file = h5py.File('data/SimFeatures.h5', 'w')
    h5file.create_dataset('particle_tracks', data = dist_array)
    h5file.close()
    return dist_array

def print_cps(cps):
    for i in range(len(cps)):
        print("Feature %3d: " % i, cps[i])

# cps = get_contact_pairs()
cps = load_contact_pairs()
