# wellVAE.py - Oliver Tsang, July 2019
#
# Trains a Variational Autoencoder (VAE) to identify the axes of a BD simulation that contain more
# signal than noise in relation to the others.
#   Run this file as the main script (probably with "python3 wellVAE.py" or "python3 -i wellVAE.py")
#
#   Uses a network built with PyTorch: two parallel encoding networks -- one to store mean values
# and one to store log-variances of the probability distribution in the latent space. The restriction
# on how much it saves is the n_z variable (number of dimensions for latent variable z).
#   There's a single decoding network that pulls from the probability distribution and tries to
# recreate the original input.
#
# This file comes with several functions to help visualize the predictions and see how well it's doing.
#
# See https://arxiv.org/pdf/1606.05908.pdf (Carl Doersch's Tutorial on Variational Autoencoders)
#     https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
#     https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
# The latter two are tutorials on training a VAE to encode handwritten digits in the MNIST database

# Dependencies: NumPy, Matplotlib, PyTorch, Scikit-Learn, and h5py (through loader.py)

print("Loading...")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb
import time
from sklearn.decomposition import PCA

###  Initializing data set  ###

# Can also load raw data or data pre-processed by PCA
from loader import sim_data as sim_data # normalized dimensions
# from loader import raw_sim_data as sim_data
# from loader import pca_sim_data as sim_data

print("... finished loading!")

###  Network Hyperparameters  ###
n_epochs = 20    # number of times to loop through the data set
batch_size = 400 # batch size
    # Minibatching (computing the gradient from multiple examples) helps with performance and stability,
    # but using smaller batches seems to help get out of local minima faster.
print("Batch size:", batch_size)

## Network dimensions
in_dim = 5 # input dimension
hsize  = 4 # size of hidden layers
n_z    = 3 # dimensionality of latent space

## Learning Algorithm Hyperparameters
optim_fn = optim.Adam
# optim_fn = optim.SGD
    # different optimization algorithms -- Adam tries to adaptively change momentum (memory of
    # changes from the last update) and has different learning rates for each parameter.
    # But standard Stochastic Gradient Descent is supposed to generalize better...

lr = 1e-2           # learning rate
weight_decay = 1e-3 # weight decay -- how much of a penalty to give to the magnitude of network weights (idea being that it's
    # easier to get less general results if the network weights are too big and mostly cancel each other out (but don't quite))
momentum = 1e-4     # momentum -- only does anything if SGD is selected because Adam does its own stuff with momentum.

kl_lambda = 0.5 # How much weight to give to the KL-Divergence term in loss?
    # Setting to 0 makes the VAE cloesr to a standard autoencoder and makes the probability distributions less smooth.
    # Changing this is as if we had chosen a sigma differently for (pred - truth)**2 / sigma**2, but parameterized differently.
    # See Doersch's tutorial on autoencoders, pg 14 (https://arxiv.org/pdf/1606.05908.pdf) for his comment on regularization.

# DataLoaders to help with minibatching
# Note that num_workers runs the load in parallel
train_loader = DataLoader(sim_data, batch_size = batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(sim_data, batch_size = 400, shuffle = False, num_workers = 4)

# helps with consistency of type -- alternative is torch.double
data_type = torch.float

# holds both encoder and decoder
class VAE(nn.Module):
    def __init__(self):
        """Set up the variational autoencoder """
        super(VAE, self).__init__() # initialization inherited from nn.Module
        self.always_random_sample = False # for the sampling process of latent variables

        # set up the encoder
        # one for means, one for variance -- there's no reason they should share layers
        self.encode_layers_means = [nn.Linear(in_dim, n_z) # just linear combination
                                   ]
        self.encode_layers_vars = [nn.Linear(in_dim, hsize),
                                   nn.ReLU(),
                                   nn.Linear(hsize, n_z)
                                  ]
        self.encode_net_means = nn.Sequential(*self.encode_layers_means)
        self.encode_net_vars  = nn.Sequential(*self.encode_layers_vars)


        # set up the decoder
        # only take in n_z because we sample from the probability distribution
        self.decode_layers = [nn.Linear(n_z, hsize),
                              nn.ReLU(),
                              nn.Linear(hsize,in_dim)
                             ]
        self.decode_net = nn.Sequential(*self.decode_layers)

        # Ensure the networks have the right data types
        # PyTorch can be touchy if some variables are floats and others are doubles
        if data_type == torch.float:
            self.encode_net_means = self.encode_net_means.float()
            self.encode_net_vars  = self.encode_net_vars.float()
            self.decode_net       = self.decode_net.float()
        else:
            self.encode_net_means = self.encode_net_means.double()
            self.encode_net_vars  = self.encode_net_vars.double()
            self.decode_net       = self.decode_net.double()

    def encode(self, x):
        """Encoding from x (input) to z (latent). 
        Returns the probability distribution in terms of mean and log-variance of a gaussian"""
        self.x = x
        self.mu     = self.encode_net_means(x)
        self.log_var = self.encode_net_vars(x)
        self.z = (self.mu, self.log_var)
        return self.z

    def sample(self, mu, log_var):
        """Does the sampling outside of the backpropagation (see the reparameterization trick)"""
        # In training mode, we want to carry the propagation through
        # otherwise, we just return the maximum likelihood mu
        # Can always change this in the __init__ function
        res = mu
        variational = True # False basically sets it to autoencoder status (but not quite b/c of the loss function)
        if variational and self.training or self.always_random_sample:
            eps_shape = (batch_size, n_z) if log_var.dim == 2 else n_z
            eps = torch.tensor(np.random.normal(0, 1, eps_shape), dtype=data_type)
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
        """Calls the other functions to run a batch/data set through the proper networks"""
        z_dist    = self.encode(x)        # 1. Encode
        z_sample  = self.sample(*z_dist)  # 2. Sample
        x_decoded = self.decode(z_sample) # 3. Decode
        # Decoding returns the max-likelihood option rather than a full prob dist
        # which seems to be desired behavior.. There's an implicit Gaussian centered around
        # it, which is where we get the square-loss for reconstruction error... see Doersch...
        return x_decoded


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
        # kl_div   = 0.5 * torch.sum(torch.exp(self.log_var)+self.mu*self.mu-1-self.log_var, dim=1)
        kl_div   = 0.5 * torch.sum(torch.exp(self.log_var)+self.mu*self.mu-1-self.log_var)
        return (self.rec_loss + kl_lambda*kl_div)/batch_size

    def run_data(self):
        """Runs the whole dataset through the network"""
        self.eval()
        data = torch.tensor(sim_data.data[:], dtype=data_type)
        recon = self(data).detach().numpy()
        return recon

    def plot_test(self, plot = True, axes=(3,3), ret = False):
        """Plots the predictions of the model and its latent variables"""
        bigdata = torch.tensor(sim_data.data[:])
        outputs = self(bigdata).detach().numpy()
        bigdata = bigdata.detach().numpy()
        latents = self.z[0].detach().numpy() # just give the means...
        if plot:
            plt.subplot(*axes, 1)
            plt.title("Latent variable%s (%d)" % (("s" if n_z>1 else ""), n_z))
            plt.plot(latents)
            for i in range(in_dim):
                plt.subplot(*axes, 2+i)
                plt.title("Particle %d" % (i+1))
                # pdb.set_trace()
                plt.plot(bigdata[:,i], label = ("Input %d"% (i+1)))
                plt.plot(outputs[:,i], label = ("Reconstruction %d" % (i+1)))
                plt.legend()
            plt.show()
        if ret:
            return outputs, latents

###  train the network!  ###
# run one epoch
def trainer(model, optimizer, epoch, models, loss_array):
    epoch_fail = 0 # loss for the epoch # credit to xkcd for the variable name()

    # run through the mini batches!
    batch_count = 0 # Keep track of how many batches we've run through
    for b_idx, data in enumerate(train_loader):
        model.train() # set training mode
        optimizer.zero_grad()
        recon = model(data)
        loss  = model.vae_loss(recon, data)
        loss_scalar = loss.item()
        rec_loss = model.rec_loss.item()/batch_size # reconstruction loss
        epoch_fail += rec_loss
        loss.backward()  # Ask PyTorch to differentiate the loss w.r.t. parameters in the model
        optimizer.step() # Ask Adam/SGD to make updates to our parameters
        batch_count += 1
        if b_idx % 100 == 0:
            print("Train Epoch %d, \tBatch index %d:   \tLoss: %f\tRecLoss: %f" % (epoch, b_idx, loss_scalar, rec_loss))
    # outputs = model(torch.tensor(sim_data.data[:])).detach().numpy()
    # epoch_loss = ((outputs - sim_data.data)**2).sum(1).mean()
    print("Epoch %d has an average reconstruction loss of  %f" % (epoch, epoch_fail/batch_count))
    # not accurate though because this isn't looking at the reconstruction loss at the end of the epoch...
    # but recomputing everything gives a avery unstable output
    models.append(vae_model)
    loss_array.append(epoch_fail/batch_count)

if __name__ == "__main__":
    # if statement is to make sure the networks can be imported to other files without this code running
    # However, if you want access to the variables by command line while running the script in interactive
    # mode, you may want to move everything outside the if statemnt.
    pass

# Compare against a naive guess of averages
naive = sim_data[-50000:].mean(0)
naive_loss = ((naive - sim_data.data[:])**2).sum(1).mean()
print("A naive guess from taking averages of the last 50000 positions yields a loss of", naive_loss)

# Compare against PCA
pca_start = time.time()
pca = PCA()
pca_latent = pca.fit_transform(sim_data.data[:])
pca_latent[:,n_z:] = 0
pca_guesses = pca.inverse_transform(pca_latent)
pca_loss = ((pca_guesses - sim_data.data[:])**2).sum(1).mean()
print("PCA gets a reconstruction loss of %f in %f seconds" % (pca_loss, time.time()-pca_start))

# Finish initializing the model
vae_model = VAE()
if optim_fn == optim.SGD:
    optimizer = optim_fn(vae_model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
else:
    optimizer = optim_fn(vae_model.parameters(), lr = lr, weight_decay = weight_decay)
models = [] # Stores old models in case an older one did better
loss_array = []
start_time = time.time() # Keep track of run time

# Now actually do the training
for epoch in range(n_epochs):
    trainer(vae_model, optimizer, epoch, models, loss_array)
    duration = time.time() - start_time
    print("%f seconds have elapsed since the training began\n" % duration)
    # validate(epoch) # to tell you how it's doing on the test set
