# runVAE.py - Oliver Tsang, July 2019
#
# main python file and is in charge of training the variational autoencoder described in VAE.py
# This file can probably be run as "python3 runVAE.py" or "python3 -i runVAE.py"
#
#     Trains a Time-lagged Autoencoder (TAE) or Variational autoencoder (VAE) to identify which
# coordinates in a BD simulation contain the most signal (and tries to find some nonlinear
# transformation to make the most of the n_z latent variables to help in later reconstruction).
#     The TAE tries to predict a future configuration (dt time steps in the future -- controlled
# by loader.py at the moment for simplicity but may be moved to this file in the future)
#     The VAE shrinks down a configuration to a predetermined number of latent variables and
# attempts to reconstruct the original configuration. The VAE is different from the standard
# autoencoder in that it models the representation in latent space by a probability distribution
# rather than as a single value. This makes the latent space a bit more interpretable and enforces
# some continuity in the relation between observable and latent space.
#     The actual TAE/VAE structure is described in VAE.py, but almost all the relevant tunable
# variables/hyperparameters such as network shape and learning rate are found in this file. This
# system is built off of PyTorch.
#
# See https://arxiv.org/pdf/1606.05908.pdf (Carl Doersch's Tutorial on Variational Autoencoders)
#     https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
#     https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
# The latter two are tutorials on training a VAE to encode handwritten digits in the MNIST database
#
# Note that the simulation data comes from sim.py and is loaded and pre-processed by loader.py
#
# Dependencies: NumPy, Matplotlib, PyTorch, Scikit-Learn, and h5py (through loader.py)

print("Loading...")
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

# import from other modules in the folder
from VAE import VAE
from loader import nor_sim_data # normalized dimensions
from loader import raw_sim_data
from loader import pca_sim_data
from loader import tla_sim_data
from loader import dt           # time lag used in dt
print("... finished loading!")

data_type = torch.float

###  Network Hyperparameters  ###
time_lagged = False

if time_lagged:
    sim_data = tla_sim_data
else:
    sim_data = nor_sim_data

n_epochs = 30
batch_size = 160

# Network dimensions
# in_dim = 8 # input dimension
in_dim = sim_data.data[:].shape[1]
if time_lagged:
    in_dim //= 2

hsize  = 6 # size of hidden layers -- don't have to all be the same size though!
n_z    = 2 # dimensionality of latent space

# the layers themselves
encode_layers_means = [nn.Linear(in_dim, hsize),
                       nn.ReLU(),
                       nn.Linear(hsize, n_z),
                       nn.ReLU(),
                       nn.Linear(n_z, n_z) # just linear combination without activation
                      ]
encode_layers_vars  = [nn.Linear(in_dim, hsize),
                       nn.ReLU(),
                       nn.Linear(hsize, n_z),
                       nn.ReLU(),
                       nn.Linear(n_z, n_z)
                      ]
decode_layers       = [nn.Linear(n_z, hsize),
                       nn.ReLU(),
                       nn.Linear(hsize, hsize),
                       nn.ReLU(),
                       nn.Linear(hsize, hsize),
                       nn.ReLU(),
                       nn.Linear(hsize,in_dim)
                      ]



## Learning Algorithm Hyperparameters
optim_fn = optim.Adam
# optim_fn = optim.SGD
    # different optimization algorithms -- Adam tries to adaptively change momentum (memory of
    # changes from the last update) and has different learning rates for each parameter.
    # But standard Stochastic Gradient Descent is supposed to generalize better...
lr = 1e-2           # learning rate
weight_decay = 1e-4 # weight decay -- how much of a penalty to give to the magnitude of network weights (idea being that it's
    # easier to get less general results if the network weights are too big and mostly cancel each other out (but don't quite))
momentum = 1e-4     # momentum -- only does anything if SGD is selected because Adam does its own stuff with momentum.

kl_lambda = 1 # How much weight to give to the KL-Divergence term in loss?
    # Setting to 0 makes the VAE cloesr to a standard autoencoder and makes the probability distributions less smooth.
    # Changing this is as if we had chosen a sigma differently for (pred - truth)**2 / sigma**2, but parameterized differently.
    # See Doersch's tutorial on autoencoders, pg 14 (https://arxiv.org/pdf/1606.05908.pdf) for his comment on regularization.

# DataLoaders to help with minibatching
# Note that num_workers runs the load in parallel
train_loader = DataLoader(sim_data, batch_size = batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(sim_data, batch_size = 400, shuffle = False, num_workers = 4)



###  train the network!  ###
def trainer(model, optimizer, epoch, models, loss_array):
    """Train for one epoch

    Arguments:
        model     (VAE): the model to be run and trained
        optimizer (fxn): usually optim.Adam or optim.SGD
        epoch     (int): the number of the current epoch
        models  (list of VAEs): array to hold old models
        loss_array (list of floats): list of loss values
    """
    epoch_fail = 0 # loss for the epoch # credit to xkcd for the variable name

    # run through the mini batches!
    batch_count = 0 # Keep track of how many batches we've run through
    for b_idx, all_data in enumerate(train_loader):
        model.train() # set training mode
        optimizer.zero_grad()

        # split up the data for the time-lagged autoencoder
        train_data = all_data if not time_lagged else all_data[:,:in_dim]
        goal_data  = all_data if not time_lagged else all_data[:,in_dim:]

        # run the model
        recon = model(train_data)

        # get loss
        loss  = model.vae_loss(recon, goal_data)
        loss_scalar = loss.item()
        rec_loss = model.rec_loss.item()/train_data.shape[0] # reconstruction loss
        epoch_fail += rec_loss

        # backprop + take a step
        loss.backward()  # Ask PyTorch to differentiate the loss w.r.t. parameters in the model
        optimizer.step() # Ask Adam/SGD to make updates to our parameters

        # print out stuff
        batch_count += 1
        if b_idx % 100 == 0:
            print("Train Epoch %d, \tBatch index %d:   \tLoss: %f\tRecLoss: %f" % (epoch, b_idx, loss_scalar, rec_loss))
    print("Epoch %d has an average reconstruction loss of  %f" % (epoch, epoch_fail/batch_count))
    # different from the reconstruction loss for the most up-to-date model
    # epoch_loss = ((outputs - sim_data.data)**2).sum(1).mean() # this is pretty unstable for some reason
    models.append(vae_model) # save it in case the loss goes up later
    loss_array.append(epoch_fail/batch_count) # keep track of loss




if __name__ == "__main__":
    # if statement is to make sure the networks can be imported to other files without this code running
    # However, if you want access to the variables by command line while running the script in interactive
    # mode, you may want to move everything outside the if statemnt.
    # pass
    # Compare against a naive guess of averages
    naive = sim_data[-50000:,:in_dim].mean(0)
    if time_lagged:
        naive_loss = ((naive - sim_data.data[:,in_dim:])**2).sum(1).mean()
    else:
        naive_loss = ((naive - sim_data.data)**2).sum(1).mean()
    print("A naive guess from taking averages of the last 50000 positions yields a loss of", naive_loss)
    
    # Compare against PCA
    pca_start = time.time()
    pca = PCA()
    pca_latent = pca.fit_transform(sim_data.data[:,:in_dim])
    pca_latent[:,n_z:] = 0
    pca_guesses = pca.inverse_transform(pca_latent)
    if time_lagged:
        pca_loss = ((pca_guesses - sim_data.data[:,in_dim:])**2).sum(1).mean()
    else:
        pca_loss = ((pca_guesses - sim_data.data[:])**2).sum(1).mean()
    print("PCA gets a reconstruction loss of %f in %f seconds" % (pca_loss, time.time()-pca_start))

# Initialize the model
vae_model = VAE(in_dim              = in_dim,
                n_z                 = n_z,
                encode_layers_means = encode_layers_means,
                encode_layers_vars  = encode_layers_vars,
                decode_layers       = decode_layers,
                variational         = True,
                kl_lambda           = kl_lambda,
                data_type           = data_type,
                pref_dataset        = sim_data.data[:])

# set the learning function
if optim_fn == optim.SGD:
    optimizer = optim_fn(vae_model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
else:
    optimizer = optim_fn(vae_model.parameters(), lr = lr, weight_decay = weight_decay)

# miscellaneous book-keeping variables
models = [] # Stores old models in case an older one did better
loss_array = []
start_time = time.time() # Keep track of run time

# Now actually do the training
for epoch in range(n_epochs):
    trainer(vae_model, optimizer, epoch, models, loss_array)
    duration = time.time() - start_time
    print("%f seconds have elapsed since the training began\n" % duration)
    # validate(epoch) # to tell you how it's doing on the test set
