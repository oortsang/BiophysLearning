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
                 decode_layers_means,
                 decode_layers_vars,
                 propagator_layers = None,
                 time_lagged = True,
                 variational = True,
                 pxz_var_init = 1,         # quasi-regularization term
                 discount  = 0,
                 data_type = np.float,
                 pref_dataset=None):         # for use with the testing functions
        """Set up the variational autoencoder"""
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
        self.out_lv_ests = nn.Parameter(torch.tensor([pxz_var_init]*in_dim, dtype = data_type))
        self.decode_net_means = nn.Sequential(*decode_layers_means)
        self.decode_net_vars  = nn.Sequential(*decode_layers_vars)

        self.variational  = variational
        self.time_lagged  = time_lagged
        self.pref_dataset = pref_dataset
        # quasi regularization factor...
        self.varnet_weight = torch.tensor(-1, dtype=data_type)
        self.pxz_var_init = pxz_var_init
        self.discount  = discount

        # Ensure the networks have the right data types
        # PyTorch can be touchy if some variables are floats and others are doubles
        self.data_type = data_type
        if self.data_type == torch.float:
            self.encode_net_means = self.encode_net_means.float()
            self.encode_net_vars  = self.encode_net_vars.float()
            self.decode_net_means = self.decode_net_means.float()
            self.decode_net_vars  = self.decode_net_vars.float()
            if self.time_lagged and propagator_layers is not None:
                self.propagator_net = self.propagator_net.float()
        else:
            self.encode_net_means = self.encode_net_means.double()
            self.encode_net_vars  = self.encode_net_vars.double()
            self.decode_net_means = self.decode_net_means.double()
            self.decode_net_vars  = self.decode_net_vars.double()
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
        output_mu = self.decode_net_means(z)
        output_lv = self.out_lv_ests  + torch.clamp(self.varnet_weight, 0,1) * self.decode_net_vars(z)
        self.xp = (output_mu, output_lv)
        return self.xp
        # tmp = self.prelim_net(z)
        # output_mu = self.decode_net_means(tmp)
        # output_lv = np.log(self.pxz_var_init) + self.decode_net_vars(tmp)
        # self.xp = (output_mu, output_lv)
        # return self.xp

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
        xp_dist   = self.decode(z_sample) # 3. Decode
        x_decoded = self.sample(*xp_dist) # 4. Sample

        if self.always_random_sample and not self.training:
            xp_dist = (x_decoded, xp_dist[1])

        # import pdb; pdb.set_trace()
        if not self.time_lagged or self.propagator_net is None:
            return (0, xp_dist[0][np.newaxis], xp_dist[1][np.newaxis])
            # Decoding returns the max-likelihood option rather than a full prob dist
            # which seems to be desired behavior.. There's a Gaussian centered around
            # it, which is where we get the square-loss for reconstruction error... see Doersch...
        else:
            fut_steps = 1
            self.z_fut = z_sample

            all_means = torch.zeros((1+fut_steps, *(xp_dist[0].shape)), dtype=self.data_type)
            all_lvs   = torch.zeros((1+fut_steps, *(xp_dist[1].shape)), dtype=self.data_type)
            all_means[0], all_lvs[0] = xp_dist

            # propagate in latent space and decode
            self.z_fut = self.propagator_net(self.z_fut)
            all_means[1+0], all_lvs[1+0] = self.decode(self.z_fut)
            return (fut_steps, all_means, all_lvs)


    def vae_loss(self, pred_means, pred_lvs, truth):
        # -log p(x) = - E[log p(x|z)] + KL[q(z)||p(z)]
        #
        # E[log P (x | z)] - Reconstruction part
        # p(x|z) = Normal(D_m(z), D_s(s))
        #
        # -log p(x|z) = (x-Dm(z))**2/(2Ds(z)) + log Ds(z) + 0.5*log(2 pi)

        if not self.variational:
            self.rec_loss = torch.sum((pred - truth)**2) # maybe replace with a pytorch function
        else:
            # self.rec_loss = 0.5*(pred_lvs+torch.exp(-pred_lvs)*(pred_means-truth)**2).sum(1)
            self.rec_loss = 0.5*(np.log(2*np.pi) + pred_lvs
                                 + torch.exp(-pred_lvs)*(pred_means-truth)**2).sum()
            # self.rec_loss = self.rec_loss.sum()

        # KL[q(z|x) || p(z)] - Kullback-Leibler divergence
        # E [ log(q(z|x)) - log(p(z)) ] using Q for the weighting on the expected value
        # p(z) = Normal(0, 1)
        # in Information-Theoretic terms, the expected gain in information/description length
        # when pulling from probability distribution P instead of Q (expectation taken over Q)
        # Equation from https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
        # but probably in some paper (2-3 lines from Doersch + diagonal covariance matrix assn.)
        kl_div = 0
        if self.variational:
            kl_div = 0.5 * (self.mu*self.mu - 1 - self.log_var + torch.exp(self.log_var)).sum()
            # kl_div = kl_div.sum()

        # return (self.rec_loss + self.pxz_var_init*kl_div) / pred_means.shape[0]

        # # Regularization term to punish latent variables from being too far
        # if self.time_lagged and self.propagator_net is not None:
        #     reg_loss = ((self.z_fut - self.encode(truth)[0])**2).sum()
        # else:
        #     reg_loss = 0
        # return (self.rec_loss + kl_div + 0.01 * reg_loss) / pred_means.shape[0]
        return (self.rec_loss + kl_div) / pred_means.shape[0]

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
        data = torch.tensor(data, dtype=self.data_type)
        _, recon, _ = self(data)
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
        _, outputs, _ = self(bigdata)
        outputs = outputs[0].detach().numpy()
        bigdata = bigdata.detach().numpy()
        latents = self.z[0].detach().numpy() # just give the means...
        if plot:
            plt.subplot(*axes, 1)
            plt.title("Latent variable%s (%d)" % (("s" if self.n_z>1 else ""), self.n_z))
            plt.suptitle("%s%sAutoencoder%s with decoder prob dist" % \
                         ("Variational " if self.variational else "",
                          "Time-lagged " if self.time_lagged else "",
                          " (w/explicit propagation)" if (self.propagator_net is not None
                                                          and self.time_lagged) else ""
                          # self.pxz_var_init
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

    def latent_plot2d(self, mode='reconstruction', data=None, bins = 60, save_name = None, show = True):
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
        # grid = np.transpose([np.tile(x_pts, y_pts.shape[0]), np.repeat(y_pts, x_pts.shape[0])])
        grid = np.transpose([np.tile(y_pts, x_pts.shape[0]), np.repeat(x_pts, y_pts.shape[0])])
        grid = torch.tensor(grid, dtype = self.data_type)

        overlay_color = np.array((1,0.65, 0, 0))*0.7 # orange
        overlay_cloud = np.array((1,0.6,0.1, 0))*0.5 # also orange
        # overlay_cloud = np.array((0,0.2,0.13, 0))* 1.75 # green
        # overlay_color = np.array((1,0.56, 0, 0))*0.65
        # inv_overlay_color = np.array((1-0.65,1-0.56*0.65, 1, 0))

        if mode == 'latent dist' or mode == 'd':
            # plot the distribution of latent variables in latent space
            latents = self.just_encode(torch.tensor(data, dtype=self.data_type)).flatten() # get the mean score for each point on the grid
            Hlat, xedges = np.histogram(latents, bins=bins)
            xpts = 0.5*(xedges[1:] + xedges[:-1])
            plt.plot(xpts, Hlat)
            if save_name is not None:
                plt.savefig(save_name)
            if show:
                plt.show()
        elif mode == 'latent potential well' or mode == 'w':
            # plot potential well in latent space
            latents = self.just_encode(torch.tensor(data, dtype=self.data_type)).flatten() # get the mean score for each point on the grid
            Hlat, xedges = np.histogram(latents, bins=bins)
            xpts = 0.5*(xedges[1:] + xedges[:-1])
            plt.plot(xpts, -np.log(Hlat+0.001)) # put a little cushion in case of a 0
            if save_name is not None:
                plt.savefig(save_name)
            if show:
                plt.show()
        elif mode == 'latent val' or mode == 'l':
            # get the mean score for each point on the grid
            latents = self.just_encode(grid).reshape(bins, bins)
            latents -= latents.min()
            latents /= latents.max()
            pic = latents[..., np.newaxis] * overlay_color *2
            pic[..., 3] = 1
            plt.imshow(pic)
            if save_name is not None:
                plt.savefig(save_name)
            if show:
                plt.show()
        elif mode == 'background' or mode == 'b':
            plt.imshow(H, cmap = 'jet', alpha = 1)
            if save_name is not None:
                plt.savefig(save_name)
            if show:
                plt.show()
        else:
            # initialize background image
            plt.imshow(H, cmap = 'jet', alpha = 0.67)
            pic = np.zeros((*H.shape, 4))
            opacity = 0.7

            # plot the requested features
            if mode == 'reconstruction' or mode == 'r':
                # attempts to reconstruct the original data
                prev_sample = self.always_random_sample
                prev_train = self.training
                self.always_random_sample = False
                self.eval()

                _, rec_points, _ = self(torch.tensor(data, dtype=self.data_type))
                rec_points = rec_points[0].clone().detach().numpy()

                H_rec, _, _ = np.histogram2d(rec_points[:,0], rec_points[:,1], bins=(x_edges,y_edges))
                H_rec -= H_rec.min()
                H_rec /= 6 * H_rec.std()
                pic = np.clip(H_rec[..., np.newaxis], 0, 1) * overlay_color
                vis = H_rec != 0
                pic[vis, 3] = np.clip(opacity, 0, 1)

                self.always_random_sample = prev_sample
                self.training = prev_train

            elif mode == 'random reconstruction' or mode == 'rr':
                # plt.close()
                # attempts to reconstruct the original data
                prev_sample = self.always_random_sample
                prev_train = self.training
                self.always_random_sample = True
                self.train()
                
                # Get a cloud of points
                _, rec_points, _ = self(torch.tensor(data, dtype=self.data_type))
                rec_points = rec_points[0].clone().detach().numpy()

                H_rec, _, _ = np.histogram2d(rec_points[:,0], rec_points[:,1], bins=(x_edges,y_edges))
                H_rec -= H_rec.min()
                H_rec /= 6 * H_rec.std()
                H_clipped = np.clip(H_rec[..., np.newaxis], 0, 1)
                vis = H_rec >= 0.05 * H_rec.std()
                pic = H_clipped * overlay_cloud

                # manage opacity
                pic[vis, 3] = np.sqrt(np.clip(H_rec[vis], 0, 0.85))

                # Get the lines
                self.always_random_sample = False
                self.eval()
                _, rec_points, _ = self(torch.tensor(data, dtype=self.data_type))
                rec_points = rec_points[0].clone().detach().numpy()
                H_rec, _, _ = np.histogram2d(rec_points[:,0], rec_points[:,1], bins=(x_edges,y_edges))
                H_rec -= H_rec.min()
                H_rec /= 6 * H_rec.std()
                line = np.clip(H_rec[..., np.newaxis] * 1, 0, 1) * overlay_color
                vis = H_rec != 0
                pic[vis,:3] = line[vis,:3]
                pic[vis, 3] = 0.85
                pic = np.clip(pic, 0, 1)
                
                # reset state
                self.always_random_sample = prev_sample
                self.training = prev_train

            elif mode == 'grid rep' or mode == 'g':
                # shows the projection of the space onto the latent space
                # _, grid_points = self(torch.tensor(grid, dtype=self.data_type))
                _, grid_points, _ = self(grid)
                grid_points = grid_points[0].clone().detach().numpy()
                H_grid, _, _ = np.histogram2d(grid_points[:,0], grid_points[:,1], bins=(x_edges,y_edges))
                H_grid -= H_grid.min()
                H_grid /= 6 * H_grid.std()
                pic = np.clip(H_grid[..., np.newaxis] * overlay_color, 0, 1)
                vis = H_grid != 0
                pic[vis, 3] = np.clip(opacity, 0, 1)
            plt.imshow(pic)
            if save_name is not None:
                plt.savefig(save_name)
            if show:
                plt.show()
