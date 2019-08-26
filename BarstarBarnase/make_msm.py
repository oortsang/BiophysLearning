from __future__ import division
import matplotlib
# matplotlib.use('Agg')
import mdtraj as md
import pyemma
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import numpy as np
from numpy import zeros, sqrt, square, log, where, pi, mean, average, arange, sin, cos
from textwrap import wrap
from cycler import cycler
from glob import glob
import sys
from itertools import groupby

def check():
    """
    Return distance in VMD (in Anstroms) with 'measure bond {atom1 atom2}'.
    measure bond {0 110} = 36.44 Angstroms = 3.644 nm = d(0,110).
    """
    top = '../go_bnbs_complex.pdb'
    ref = md.load('../go_bnbs_complex.pdb')
    traj_dir = '../trajectories_stride1000/'
    print
    pairs = []
    cutoff_distance = 6    # nanometers
    for i in range(1):
        t = md.load(top)
        for bn in xrange(0,1):
            for bs in xrange(110,111):
                d = md.compute_distances(t, ((bn,bs),))
                print d    # nanometers
    t = md.load(traj_dir, top=top)
    print t

def contacts():
    """
    Return contact pairs between the 100 Barnase beads and 89 barstar beads
    for featurization with TICA.  Only keep pairs within a cutoff_distance and 
    lasting longer than 10 frames (1 ns).
    """
    # Load trajectories
    cutoff_distance = 1.2  # nanometers
    label = int(cutoff_distance*10)
    top = '../go_bnbs_complex.pdb'
    traj_dir = '../trajectories_stride1000/'
    print cutoff_distance, '\n'

    # Loop over 25 aggregate trajectories, 110 Barnase beads, 89 Barstar beads
    pairs = []
    for i in xrange(25):
        print '\nTrajectory', i
        t = md.load_dcd(traj_dir+'1brs_cat_{}.dcd'.format(i), top=top)
        for bn in range(0,110):
            for bs in range(110,199):
                count = []
                contiguous = []
                d = md.compute_distances(t, ((bn,bs),))
                print 'BN, BS = ', bn, ',', bs # , '\n', t, '\n', t.xyz.shape
                # Count: contact distances < cutoff_distance as 1
                #        contact distances > cutoff_distance as 0
                for c, el in enumerate(d):
                    if el < cutoff_distance:
                        count.append(1)
                    else:
                        count.append(0)

                # Distances < cutoff_distance and lasting longer than 10 steps
                # Use itertools.groupby to count contiguous elements of 0 or 1
                grouped_count = [[k, sum(1 for i in g)] for k, g in groupby(count)]
                grouped_count = np.array(grouped_count)
                for c, el in enumerate(grouped_count):
                    if el[0] == 1:
                        contiguous.append(el[1])
                    else:
                        contiguous.append(el[0])
                print 'Longest contact time = ', max(contiguous), 'frames'
                if max(contiguous) > 10:
                    pairs.append([bn,bs])
                sys.stdout.flush()
    print '\npairs = ', pairs, '\n', len(pairs)
    np.save('contact_pairs/contact_pairs_12A_all'.format(i), pairs)
    return pairs

def group_pairs(cut):
    """
    Return unique contact pairs between BN, BS and their number of occurences. 
    Contact pairs are those within a cutoff distance and lasting longer than
    10 frames (1 ns).
    """
    cut = 12
    p = np.load('contact_pairs/contact_pairs_{}A_all.npy'.format(cut))
    grouped_p = {}
    list_p = []
    for i,j in p:
        count = 0
        for k in range(len(p)):
            if i == p[k][0]:
                if j == p[k][1]:
                    count += 1
            else:
                count = count
        grouped_p[i,j] = count
    unique_pairs = np.array(sorted(grouped_p.items()))
    print 'Contact pairs and occurences = \n', unique_pairs
    print 'Number of unique contact pairs <', cut,\
          'A and bound longer than 1 ns = ', len(unique_pairs)
    np.save('unique_pairs', unique_pairs)
    return unique_pairs

def tica():
    """
    Run TICA using features defined by contact pairs between BN, BS.
    Contact pairs are those within a cutoff distance and lasting longer than
    10 frames (1 ns).
    """
    cutoff_distance = 1.2  # nanometers
    label = int(cutoff_distance*10)
    # unique_pairs = group_pairs(label)
    unique_pairs = np.load('unique_pairs.npy')
    lag = 20
    kT = 0.596

    # Get contact pair indices and load trajectories
    indices = [unique_pairs[i][0] for i in range(len(unique_pairs))]
    print indices, '\n'
    print np.shape(indices), np.shape(unique_pairs)
    top = '../go_bnbs_complex.pdb'
    t = glob('../trajectories_stride1000/*.dcd')

    # Featurize
    feat = coor.featurizer(top)
    feat.add_distances(indices=indices)
    inp = coor.source(t, feat)
    print '\n', feat.describe(), '\n'
    print 'Feature dimension = ', feat.dimension()
    print 'Number of trajectories = ',inp.number_of_trajectories()
    print 'Trajectory length = ',inp.trajectory_length(0)
    print 'Number of dimension = ',inp.dimension()

    # Run TICA
    tica_obj = coor.tica(inp, lag=lag, dim=4, kinetic_map=False)
    Y = tica_obj.get_output()  # get coordinates in TICA space

    # for i in range(2):  # Trick to make eigenvalues look better
    #     if tica_obj.eigenvectors[0, i] > 0:
    #         tica_obj.eigenvectors[:, i] *= -1

    print 'TICA dimension = ', tica_obj.dimension()
    print 'TICA cumulative variance = \n', tica_obj.cumvar
    print 'Number of trajectories = ', np.shape(Y)[0]
    print 'Number of frames = ', np.shape(Y)[1]
    print 'Number of dimensions = ',np.shape(Y)[2]
    print 'Shape of tica data = ', np.shape(Y)
    print 
    print tica_obj.eigenvalues
    print len(tica_obj.eigenvalues)
    print tica_obj.cumvar[0:20]

    np.save('Y', Y)
    np.save('eigenvalues.npy', tica_obj.eigenvalues)
    np.save('eigenvectors.npy', tica_obj.eigenvectors)
    np.save('tica_timescales.npy', tica_obj.timescales)
    return Y, tica_obj

def get_rmsd():
    """Return the RMSD of the barnase-barstar complex."""
    top = '../go_bnbs_complex.pdb'
    t = glob('../trajectories_stride1000/*.dcd')
    traj = md.load(t, top=top)
    rmsd_complex = []
    for i in range(25):
        # t = glob('../trajectories_stride1000/*_{}.dcd'.format(i))
        t = glob('../trajectories_stride1000/*')[i]
        traj = md.load(t, top=top)   
        r = md.rmsd(traj, md.load(top), 0).reshape(-1,1)
        rmsd_complex.append(np.float64(r))
    return rmsd_complex

def combine_features():
    """
    Combine features of different shapes along axis = 2.
    NOTE: Make sure that features are in same order wrt traj number, time step
    """
    Y = list(np.load('Y.npy')[:,:,0:4]) # tica output
    rmsd = get_rmsd()
    com = np.load('com_dist.npy')  # nanometers
    com = com.reshape(len(com), len(com[0]), 1)
    f = np.concatenate((Y, rmsd, com), axis=2)
    np.save('features.npy', f)
    # f = np.concatenate((rmsd, com), axis=2)
    print 'shape of tica output =', np.shape(Y)
    print 'shape of distance =', np.shape(rmsd)
    print 'shape of com =', np.shape(com)
    print 'shape of combined features =', np.shape(f)
    return f

def cluster():
    """Clustering with k-means."""
    n_clusters = 550
    kT = 0.596

    # Load data
    top = '../go_bnbs_complex.pdb'
    t = glob('../trajectories_stride1000/*.dcd')
    traj = md.load(t, top=top)
    Y = list(np.load('Y.npy')[:,:,0:4]) # tica output
    feat = list(combine_features())

    # Clustering
    clustering = coor.cluster_kmeans(data=feat, k=n_clusters)
    dtrajs =  clustering.dtrajs
    print 'Length dtrajs = ', len(dtrajs), '\n'
    print 'dtrajs shape = ', np.shape(dtrajs), '\n'
    print 'dtrajs type = ', type(dtrajs), '\n'

    # IMPLIED TIMESCALES
    lags = 150
    nits = 100
    its = msm.timescales_msm(dtrajs, lags=lags, errors='bayes', nits=nits)
    plt.figure()
    mplt.plot_implied_timescales(its, ylog=False, dt=1.0, units='steps', linewidth=2)
    plt.savefig('figures/implied_ts.pdf')
    plt.close()
    # plt.figure()
    # mplt.plot_implied_timescales(its, ylog=True, dt=1.0, units='steps', linewidth=2)
    # plt.savefig('figures/implied_ts_log.pdf')
    np.savez('clustering_{}.npz'.format(n_clusters), describe=clustering.describe(),
            get_output=clustering.get_output(),
            get_params=clustering.get_params(),
            n_frames_total=clustering.n_frames_total(),
            number_of_trajectories=clustering.number_of_trajectories(),
            clustercenters=clustering.clustercenters,
            converged=clustering.converged,
            data_producer=clustering.data_producer,
            dtrajs=clustering.dtrajs,
            filenames=clustering.filenames,
            index_clusters=clustering.index_clusters,
            init_strategy=clustering.init_strategy)
    return dtrajs

def implied_timescales():
    """Plot implied timescales."""
    # Load data
    top = '../go_bnbs_complex.pdb'
    t = glob('../trajectories_stride1000/*.dcd')
    traj = md.load(t, top=top)
    Y = list(np.load('Y.npy')[:,:,0:4]) # tica output
    feat = list(combine_features())
    clustering = np.load('clustering_550.npz')
    dtrajs = list(clustering['dtrajs'])

    lags = 150
    nits = 100
    its = msm.timescales_msm(dtrajs, lags=lags, errors='bayes', nits=nits)
    plt.figure()
    mplt.plot_implied_timescales(its, ylog=False, dt=1.0, units='steps', linewidth=2)
    plt.savefig('figures/implied_ts.pdf')
    plt.close()
    print its.get_timescales()
    print np.shape(its.get_timescales())
    np.save('its.npy', its.get_timescales())

def build_msm():
    """Construct Markov state models."""
    Y = list(np.load('Y.npy')[:,:,0:4])
    clustering = np.load('clustering_550.npz')
    dtrajs = list(clustering['dtrajs'])
    # dtrajs = list(np.load('dtrajs.npy'))
    print 'dtrajs = \n', dtrajs, '\n'
    print 'length dtrajs = ', len(dtrajs), '\n'
    print 'dtrajs shape = ', np.shape(dtrajs), '\n'
    print 'dtrajs type = ', type(dtrajs), '\n'
    print '# clusters = ', clustering['clustercenters'].shape

    M = msm.estimate_markov_model(dtrajs, 150)
    print 'fraction of states used = ', M.active_state_fraction
    print 'fraction of counts used = ', M.active_count_fraction
    M.save('saved_msm_model.pyemma')

    # Plot timescales
    # plt.figure()
    # plt.plot(M.timescales(), linewidth=0, marker='o')
    # plt.xlabel('index'); plt.ylabel('timescale steps (0.1 ns)'); plt.xlim(-0.5,10.5)
    # plt.title('MSM timescales')
    # plt.savefig('figures/msm_timescales.png')
    # plt.figure()
    # plt.plot(M.timescales()[:-1]/M.timescales()[1:], linewidth=0, marker='o')
    # plt.xlabel('index'); plt.ylabel('timescale separation'); plt.xlim(-0.5,10.5)
    # plt.title('Ratio of subsequent MSM timescales')
    # plt.savefig('figures/msm_timescales_ratio.png')
    # plt.close()

    # Chapman-Kolmogorov Test
    # print 'ck test 2'
    # ck = M.cktest(2, mlags=50, conf=0.95, err_est=False)
    # # plt.figure()
    # mplt.plot_cktest(ck, diag=False, padding_top=0.1, y01=False, padding_between=0.3, dt=0.1, units='ns')
    # plt.savefig('figures/cktest_2_v2.png')
    # plt.show()

    # ck = M.cktest(3, conf=0.95, err_est=False)
    # plt.figure()
    # mplt.plot_cktest(ck, diag=False, padding_top=0.1, y01=False, padding_between=0.3,dt=0.1, units='ns')
    # plt.savefig('figures/cktest_3.png')

    """# PCCA Active Sets
    M.pcca(2)
    print 
    print 'MSM MFPT 1 -> 2 = %6.0f steps' % M.mfpt(M.metastable_sets[0], M.metastable_sets[1])
    print 'MSM MFPT 2 -> 1 = %6.0f steps' % M.mfpt(M.metastable_sets[1], M.metastable_sets[0])
    np.savez('mfpt_msm.npz',
        active_set=M.active_set,
        stationary_distribution=M.stationary_distribution,
        metastable_assignments=M.metastable_assignments)
    """

    """
    print 'Full count matrix = \n', M.count_matrix_full
    print 'Left eigenvectors = \n', M.eigenvectors_left()
    print 'Eigenvalues = \n', M.eigenvalues()
    print 'Stationary distribution = \n', M.stationary_distribution
    print 'Number of states = ', M.nstates
    print 'Left eigenvectors = \n', M.eigenvectors_left()
    print
    print M.eigenvectors_left()[1]
    print
    print cc_x.shape
    print M.eigenvectors_left().shape

    # PLOT MSM POPULATIONS
    Y = list(np.load('Y.npy'))
    kT = 0.596
    F = np.load('F.npy')
    cc_x = np.load('clustercenters_x.npy')
    cc_y = np.load('clustercenters_y.npy')
    z,x,y = np.histogram2d(np.vstack(Y)[:,0], np.vstack(Y)[:,1], bins=100)
    extent = [x[0], x[-1], y[-0], y[-1]]
    # contour_x = np.linspace(x[0], x[-1], 100)
    # contour_y = np.linspace(y[0], y[-1], 100)
    # contour_X, contour_Y = np.meshgrid(contour_x, contour_y)
    plt.figure()
    cmap_f = 'Blues_r'
    cmap_msm = 'coolwarm'
    # plt.contourf(contour_X, contour_Y, F.T, 100, cmap="Blues_r", extent=extent)
    plt.contourf(F.T, 100, cmap=cmap_f, extent=extent)
    plt.scatter(cc_x, cc_y,
                s=1e4 * M.stationary_distribution,       # size by population
                c=M.eigenvectors_left()[1], # color by eigenvector
                cmap=cmap_msm,
                zorder=3) 
    plt.colorbar(label='First dynamical eigenvector')
    plt.xlabel('IC 1')
    plt.ylabel('IC 2')
    plt.tight_layout()
    plt.savefig('figures/msm.pdf')
    plt.show()
    """
    print 'Job completed'

def estimate_error():
    Y = list(np.load('Y.npy')[:,:,0:4])
    clustering = np.load('clustering_550.npz')
    dtrajs = list(clustering['dtrajs'])
    dt = 100000  # Saving frequency
    lag = 120
    bmsm = msm.bayesian_markov_model(dtrajs, lag=lag, statdist=None, count_mode='sample', nsamples=100)

    # print 'Timescale uncertainty:'
    # print 'estimated sample mean =', bmsm.sample_mean('timescales', 1)
    # print 'estimated sample standard deviation =', bmsm.sample_std('timescales', 1)
    # print 'relative error of the estimate =', bmsm.sample_std('timescales', 1) / bmsm.sample_mean('timescales', 1), '\n'

    print 'Eigenvalue uncertainty:'
    print 'estimated sample mean =', bmsm.sample_mean('eigenvalues', 1)
    print 'estimated sample standard deviation =', bmsm.sample_std('eigenvalues', 1)
    print 'relative error of the estimate =', bmsm.sample_std('eigenvalues', 1) / bmsm.sample_mean('eigenvalues', 1), '\n'

    print 'Eigenvalue uncertainty:'
    print 'estimated sample mean =\n', bmsm.sample_mean('eigenvalues', 7)
    print 'estimated sample standard deviation =\n', bmsm.sample_std('eigenvalues', 7)
    print 'relative error of the estimate =\n', bmsm.sample_std('eigenvalues', 7) / bmsm.sample_mean('eigenvalues', 7), '\n'

    # print 'Stationary distribution uncertainty:'
    # print 'estimated sample mean =', bmsm.sample_mean('stationary_distribution', 1)
    # print 'estimated sample standard deviation =', bmsm.sample_std('stationary_distribution', 1)
    # print 'relative error of the estimate =', bmsm.sample_std('stationary_distribution', 1) / bmsm.sample_mean('stationary_distribution', 1), '\n'

# check()
# contacts()
# group_pairs()
# tica()
# combine_features()
# cluster()
build_msm()
# estimate_error()
