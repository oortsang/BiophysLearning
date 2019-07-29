# VAE.py - Oliver Tsang, July 2019
#
# Class to hold the Variational Autoencoder class (which doubles as a Time-lagged autoencoder)
#
# runVAE.py is in charge of the actual initialization and training on the simulation datasets.
#
#   Uses a network built with PyTorch: two parallel encoding networks -- one to store mean values
# and one to store log-variances of the probability distribution in the latent space. The restriction
# on how much it saves is the n_z variable (number of dimensions for latent variable z).
#   There's a single decoding network that pulls from the probability distribution and tries to
# recreate the original input.

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb


# holds both encoder and decoder
#
# Class VAE has the following functions:
# __init__ - initialization
# encode - first half of the forward pass
# sample - sample from the latent variable's probability distribution to feed to the decoder
# decode - tries to reconstruct the original input (or time-lagged input in the case of the TAE)
# vae_loss - computes the loss function given predictions and desired outputs
# run_data - runs through the whole data set
# plot_test - makes pretty plots of the model's predictions on the time series
# latent_plot2d - for 2D data, it plots the latent subspace on top of the distribution of real data
#                 and also plots the distribution of a 1D latent variable
#
# Note that whether it behaves like a time-lagged autoencoder or not is up to the training process
class VAE(nn.Module):
    def __init__(self,
                 in_dim,                # input dimension -- redundant with encoding layers but more convenient
                 n_z,                   # number of latent variables
                 encode_layers_means,
                 encode_layers_vars,
                 decode_layers,
                 propagator_layers = None,
                 time_lagged = True,
                 variational = True,
                 kl_lambda = 1,         # quasi-regularization term
                 discount  = 0,
                 data_type = np.float,
                 pref_dataset=None):         # for use with the testing functions
        """Set up the variational autoencoder """
        super(VAE, self).__init__() # initialization inherited from nn.Module
        self.always_random_sample = False # for the sampling process of latent variables
        # set up the encoder
        # one for means, one for variance -- there's no reason they should share layers
        self.in_dim = in_dim
        self.n_z = n_z
        self.encode_net_means = nn.Sequential(*encode_layers_means)
        self.encode_net_vars  = nn.Sequential(*encode_layers_vars)

        # set up the propagator
        if time_lagged and propagator_layers is not None:
            self.propagator_net = nn.Sequential(*propagator_layers)
        else:
            self.propagator_net = None

        # set up the decoder
        self.decode_net = nn.Sequential(*decode_layers)

        self.variational  = variational
        self.time_lagged  = time_lagged
        self.pref_dataset = pref_dataset
        # quasi regularization factor...
        self.kl_lambda = kl_lambda
        self.discount  = discount

        # Ensure the networks have the right data types
        # PyTorch can be touchy if some variables are floats and others are doubles
        self.data_type = data_type
        if self.data_type == torch.float:
            self.encode_net_means = self.encode_net_means.float()
            self.encode_net_vars  = self.encode_net_vars.float()
            self.decode_net       = self.decode_net.float()
            if self.time_lagged and propagator_layers is not None:
                self.propagator_net = self.propagator_net.float()
        else:
            self.encode_net_means = self.encode_net_means.double()
            self.encode_net_vars  = self.encode_net_vars.double()
            self.decode_net       = self.decode_net.double()
            if self.time_lagged and propagator_layers is not None:
                self.propagator_net = self.propagator_net.double()

    def encode(self, x):
        """Encoding from x (input) to z (latent).
        Returns the probability distribution in terms of mean and log-variance of a gaussian"""
        # saves mu and log_var to compute the loss
        # saves x for debugging convenience
        self.x = x
        self.mu      = self.encode_net_means(x) # simple mean
        self.log_var = self.encode_net_vars(x)  # guess the log-variance for numerical stability
        self.z = (self.mu, self.log_var)
        return self.z

    def sample(self, mu, log_var):
        """Does the sampling outside of the backpropagation (see the reparameterization trick)"""
        # In training mode, we want to carry the propagation through
        # otherwise, we just return the maximum likelihood mu
        # Can always change this in the __init__ function
        res = mu

        if (self.variational and self.training) or self.always_random_sample:
            eps_shape = log_var.shape
            eps = torch.tensor(np.random.normal(0, 1, eps_shape), dtype=self.data_type)
            # eps contains a randomly pulled number for each example in the minibatch
            res += torch.exp(log_var*0.5) * eps # elementwise multiplication is desired
        return res

    def decode(self, z):
        """Takes a single z sample and attempts to reconstruct the inputs"""
        if False and not self.training:
            # save the parameters ... for visualization purposes
            self.post_first_linear  = self.decode_layers[0](z)
            self.activated          = self.decode_layers[1](self.post_first_linear)
            self.post_second_linear = self.decode_layers[2](self.activated)
            self.final_output       = self.decode_layers[3](self.post_second_linear)
            return self.final_output
        else:
            output = self.decode_net(z)
            return output

    # run through encoder, sampler, and decoder
    def forward(self, x):
        """Calls the other functions to run a batch/data set through the proper networks.
        To deal with the possibility of propagation, we return a tuple - first element is
        is the number of future steps to expect in the tensor...
        """
        # if we feed in too many dimensions, just take the first self.in_dim
        if x.dim() == 2 and x.shape[-1] != self.in_dim:
            x = x[:, :self.in_dim]
        z_dist    = self.encode(x)        # 1. Encode
        z_sample  = self.sample(*z_dist)  # 2. Sample
        x_decoded = self.decode(z_sample) # 3. Decode
        # import pdb; pdb.set_trace()
        if not self.time_lagged or self.propagator_net is None:
            return (0, x_decoded[np.newaxis])
            # Decoding returns the max-likelihood option rather than a full prob dist
            # which seems to be desired behavior.. There's an implicit Gaussian centered around
            # it, which is where we get the square-loss for reconstruction error... see Doersch...
        else:
            fut_steps = 1
            all_xs = torch.zeros((1+fut_steps, *(x_decoded.shape)), dtype=self.data_type)
            all_xs[0] = x_decoded
            z_fut = z_sample

            # propagate in latent space and decode
            z_fut = self.propagator_net(z_fut)
            all_xs[1+0] = self.decode(z_fut)
            return (fut_steps, all_xs)


    def vae_loss(self, pred, truth):
        # E[log P (x | z)] - Reconstruction loss
        #
        # Use dimension-wise square loss since it's Gaussian
        #
        # Say that the likelihood of reconstructing x from latent variable z
        # is described by a Gaussian centered around the original x...
        self.rec_loss = torch.sum((pred - truth)**2) # maybe replace with a pytorch function...

        # KL[Q(z|x) || P(z|x)] - Kullback-Leibler divergence
        # E [ log(Q(z|x)) - log(P(z|x)) ] using Q for the weighting on the expected value
        # in Information-Theoretic terms, the expected gain in information/description length
        # when pulling from probability distribution P instead of Q (expectation taken over Q)

        # Equation from https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
        # but probably in some paper (2-3 lines from Doersch + diagonal covariance matrix assn.)
        kl_div = 0
        if self.variational:
            kl_div   = 0.5 * torch.sum(self.mu*self.mu - 1 - self.log_var + torch.exp(self.log_var))
        return (self.rec_loss + self.kl_lambda*kl_div) / pred.shape[0]

    def just_encode(self, x):
        """Runs data through just the encoder"""
        # if we feed in too many dimensions, just take the first self.in_dim
        if x.dim() == 2 and x.shape[-1] != self.in_dim:
            x = x[:, :self.in_dim]
        z_dist = self.encode(x)
        return z_dist[0].detach().numpy() # returns just the means

    def run_data(self, data=None):
        """Runs the whole dataset through the network"""
        if data is None:
            if self.pref_dataset is None:
                return None
            data = self.pref_dataset
        self.eval()
        data = torchtensor(data, dtype=self.data_type)
        _, recon = self(data)
        return recon[0].detach().numpy()

    def plot_test(self, data=None, plot = True, axes=(None,1), ret = False, dt = 0):
        """Plots the predictions of the model and its latent variables"""
        if axes[0] is None:
            axes = (1+self.in_dim, axes[1])

        if data is None:
            if self.pref_dataset is None:
                return None
            data = self.pref_dataset
        bigdata = torch.tensor(data)
        _, outputs = self(bigdata)
        outputs = outputs[0].detach().numpy()
        bigdata = bigdata.detach().numpy()
        latents = self.z[0].detach().numpy() # just give the means...
        if plot:
            plt.subplot(*axes, 1)
            plt.title("Latent variable%s (%d)" % (("s" if self.n_z>1 else ""), self.n_z))
            plt.suptitle("%s%sAutoencoder%s with $\\lambda_{KL}$ = %.2f" % \
                         ("Variational " if self.variational else "",
                          "Time-lagged " if self.time_lagged else "",
                          " (w/explicit propagation)" if (self.propagator_net is not None
                                                          and self.time_lagged) else "",
                          self.kl_lambda
                         )
            )
            plt.plot(latents)
            for i in range(self.in_dim):
                plt.subplot(*axes, 2+i)
                plt.title("Particle %d" % (i+1))
                # pdb.set_trace()
                plt.plot(bigdata[:,i], label = ("Input %d"% (i+1)))
                plt.plot(range(dt,dt+outputs.shape[0]), outputs[:,i], label = ("Reconstruction %d" % (i+1)))
                plt.legend(loc = 'lower right')
            plt.show()
        if ret:
            return outputs, latents

    def latent_plot2d(self, mode='reconstruction', data=None, bins = 30):
        """Note that this is only designed for 2D spaces

        Arguments:
            values: Nx2 array holding a sequence of coordinates
            bins:   number of bins to use when calling the histogram
        """
        # Load the proper data set
        if data is None:
            if self.pref_dataset is None:
                return None
            data = self.pref_dataset

        # construct a mesh and histogram from the particle's distribution in space
        H, x_edges, y_edges = np.histogram2d(data[:,0], data[:,1], bins = bins)
        x_pts = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_pts = 0.5 * (y_edges[:-1] + y_edges[1:])
        grid = np.transpose([np.tile(x_pts, y_pts.shape[0]), np.repeat(y_pts, x_pts.shape[0])])
        grid = torch.tensor(grid, dtype = self.data_type)

        overlay_color = np.array((1,0.56, 0, 0))*0.65

        if mode == 'latent dist' or mode == 'd':
            # plot the distribution of latent variables in latent space
            latents = self.just_encode(torch.tensor(data, dtype=self.data_type)).flatten() # get the mean score for each point on the grid
            Hlat, xedges = np.histogram(latents, bins=bins)
            xpts = 0.5*(xedges[1:] + xedges[:-1])
            plt.plot(xpts, Hlat)
            plt.show()
        elif mode == 'latent potential well' or mode == 'w':
            # plot potential well in latent space
            latents = self.just_encode(torch.tensor(data, dtype=self.data_type)).flatten() # get the mean score for each point on the grid
            Hlat, xedges = np.histogram(latents, bins=bins)
            xpts = 0.5*(xedges[1:] + xedges[:-1])
            plt.plot(xpts, -np.log(Hlat+0.001)) # put a little cushion in case of a 0
            plt.show()
        elif mode == 'latent val' or mode == 'l':
            # get the mean score for each point on the grid
            latents = self.just_encode(grid).reshape(bins, bins)
            latents -= latents.min()
            latents /= latents.max()
            pic = latents[..., np.newaxis] * overlay_color *2
            pic[..., 3] = 1
            plt.imshow(pic)
            plt.show()
        else:
            # initialize background image
            plt.imshow(H, cmap = 'jet', alpha = 0.7)
            pic = np.zeros((*H.shape, 4))
            # overlay_color = np.array((0,0,0,0.0))
            opacity = 0.7

            # plot the requested features
            if mode == 'reconstruction' or mode == 'r':
                # attempts to reconstruct the original data
                _, rec_points = self(torch.tensor(data, dtype=self.data_type))
                rec_points = rec_points[0].clone().detach().numpy()

                H_rec, _, _ = np.histogram2d(rec_points[:,0], rec_points[:,1], bins=(x_edges,y_edges))
                H_rec -= H_rec.min()
                H_rec /= H_rec.max()
                pic = H_rec[..., np.newaxis] * overlay_color
                vis = H_rec != 0
                pic[vis, 3] = opacity

            elif mode == 'grid rep' or mode == 'g':
                # shows the projection of the space onto the latent space
                # _, grid_points = self(torch.tensor(grid, dtype=self.data_type))
                _, grid_points = self(grid)
                grid_points = grid_points[0].clone().detach().numpy()
                H_grid, _, _ = np.histogram2d(grid_points[:,0], grid_points[:,1], bins=(x_edges,y_edges))
                H_grid -= H_grid.min()
                H_grid /= H_grid.max()
                pic = H_grid[..., np.newaxis] * overlay_color
                vis = H_grid != 0
                pic[vis, 3] = opacity
            plt.imshow(pic)
            plt.show()
