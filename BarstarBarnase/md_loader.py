# Oliver Tsang, Summer 2019
# (Using Python 3)
# Partly copied from Ziwei He's make_msm.py file (which is in Python 2)

import mdtraj as md
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm # for a nice progress bar
import h5py
from os.path import getmtime, isfile
import h5py

# File names
top_file  = "go_bnbs_complex.pdb"
traj_file = lambda i: "MD/1brs_cat_%d.dcd" % i
dist_file = lambda i: "data/cp_distances_%d.h5" % i
cps_file = "My Contact Pairs/cpall.npy"
output_file = "data/SimFeatures.h5"

# Initialize files
num = 2
top = md.load_pdb(top_file)
main_traj = md.load_dcd(traj_file(num), top = top)
files = [0, 1, 2, 3] # np.arange(3)

# Parameters for get_contact_pairs
cutoff_dist = 1.2 # nm
cutoff_time = 10  # timesteps = ns

# Time-lag parameter
dt = 50 # timesteps

##### Finding the contact pairs  #####
def get_contact_pairs():
    """
    Return pairs of AAs that are within the cutoff distance for a duration lasting longer
    than the cutoff time (10 frames = 10 ns). This is meant to select narrow down the num-
    ber of features we deal with later.
    """
    pairs = []
    pad = lambda nparr, length: np.concatenate((np.array([-1]), nparr, np.array([length])), 0)

    for fi in tqdm(files):
        trajf = md.load_dcd(traj_file(fi), top = top)
        for ni in range(0, 110):
            for si in range(110, 199):
                ds = md.compute_distances(trajf, ((ni,si),)).flatten()
                # Keep track of time spent outside the cutoff.. reasons become clear shortly
                outside_cutoff = ds > cutoff_dist
                out_idcs = np.where(outside_cutoff)[0]

                # find the durations between sejours outside the cutoff
                # (i.e., how many timesteps are spent within the cutoff)
                # Padding creates virtual points at -1 and the end - we say
                # these points are outside the cutoff, just to be sure.
                # Subtract 1 because we expect a difference of one in a contiguous section
                durs = np.diff(pad(out_idcs, ds.shape[0])) - 1

                if durs.max() > cutoff_time:
                    pairs.append((ni, si))
    un_pairs = np.array(pairs) # unique pairs
    un_pairs = np.unique(un_pairs, axis = 0)
    print("Found %d contact pairs out of a possible %d" % (len(un_pairs), 110*89))
    np.save(cps_file, un_pairs)
    return pairs

def load_contact_pairs(file=cps_file):
    """Loads the list of contact pairs from disk"""
    cps = np.load(cps_file)
    return cps

def print_cps(cps):
    print("Here is a list of the contact pairs found:\n")
    for i in range(len(cps)):
        print("Feature %3d: " % i, cps[i])


##### Preparing the distances array(s) for later steps #####

def create_dist_array(contact_pairs, traj = main_traj, save_name = None):
    """Creates a numpy array with the distances between contact pairs over the time series"""
    dist_array = np.zeros((1+len(contact_pairs), traj.n_frames), dtype = np.float32)
    dist_array[0] = np.arange(traj.n_frames) # Add indices to keep track of separate trajectories
    for i, p in enumerate(contact_pairs):
        dist_array[i+1] = md.compute_distances(traj, (p,)).flatten()
    dist_array = dist_array.T
    # Optionally write to disk
    if save_name is not None:
        h5file = h5py.File(save_name, 'w')
        h5file.create_dataset('particle_tracks', data = dist_array)
        h5file.close()
    return dist_array

def make_and_save_dists(contact_pairs, fi):
    """Calls create_dist_array to compute the distances for contact pairs --
    but this function runs through every file in 'files'"""
    tf_name = traj_file(fi)
    traj_i = md.load_dcd(tf_name, top = top) # trajectory no. i
    dists = create_dist_array(contact_pairs, traj_i, save_name = dist_file(fi))
    return dists

# def time_lag(data, dt):
#     """Creates a numpy array with distances between contact pairs but also includes
#     the same data from dt timesteps later"""
#     # data = create_array(contact_pairs, traj)
#     lag_data = np.zeros((data.shape[0]-dt, 2*data.shape[1]), dtype = np.float32)
#     lag_data[:, : data.shape[1]] = data[: -dt, :]
#     lag_data[:, data.shape[1] :] = data[ dt :, :]
#     return lag_data


def merge_arrays(contact_pairs, dt, save_file = output_file):
    """Takes the intermediate individual non-time-lagged files with distances between
    contact pairs and merges them together (while applying the time lag)"""
    # Loop through each file in "files"
    # 1. Check to see whether the h5 files are outdated vs the contact_pairs files.
    #   a. If so, recalculate using make_and_save_dists for everything
    #   b. Otherwise, just load them and keep going
    # 2. Straight-forward merger...
    #   a. Apply time_lag function
    #   b. Maybe save to file after each iteration in case of crashing?
    dists_so_far = None
    for fi in tqdm(files):
        # fi_name = traj_file(fi)
        fi_name = dist_file(fi)
        if (not isfile(fi_name)) or (getmtime(fi_name) < getmtime(cps_file)):
            # (re)compute
            dists = make_and_save_dists(contact_pairs, fi)
        else:
            # load
            h5file = h5py.File(fi_name, 'r')
            dists = h5file['particle_tracks'][:]
            h5file.close()
        # # tl_data = time_lag(dists, dt) # don't do this for now
        # tl_data = dists
        # if dists_so_far is None:
        #     dists_so_far = tl_data
        if dists_so_far is None:
            dists_so_far = dists
        else:
            dists_so_far = np.concatenate([dists_so_far, dists])
    h5file = h5py.File(save_file, 'w')
    h5file.create_dataset('particle_tracks', data = dists_so_far)
    h5file.close()
    return dists_so_far


if __name__ == "__main__":
    if False:
        # cps = get_contact_pairs()
        # if input("Save these? y/n: ")[0].lower() == 'y':
        #     create_array(cps, traj)
        #     print("Saved!")
        pass
    cps = load_contact_pairs()

    merged = merge_arrays(cps, dt)
