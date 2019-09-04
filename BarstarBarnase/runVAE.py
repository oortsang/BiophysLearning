#!/usr/bin/python
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
# rather than as a single value. This makes the latent space a bit more interpretable and
# enforces some continuity in the relation between observable and latent space.
#     The actual TAE/VAE structure is described in VAE.py, but almost all the relevant tunable
# variables/hyperparameters such as network shape and learning rate are found in this file. This
# system is built off of PyTorch.
#
#     This version of the code supports an "explicit propagator." In the TAE described above,
# the input (x_t) is at time t, and the outputted "reconstruction" is at time t+dt (x'_(t+dt)).
# However, this means that the meaning of the latent space is a bit ambiguous, and it's
# unclear where the time step into the future is happening.
#     Therefore, in the explicit propagator model, the latent variable at time t (= z_t) gets
# passed a decoder to reconstruct at time t (= x'_t). Meanwhile, we also run the latent variable
# through a propagator to get z_(t+dt) = P (z_t). This new latent variable z_(t+dt) is then
# passed through the decoder to find a reconstruction for time t+dt (= z_(t+dt)).



# variable representing time t gets copied. The original, z_t is passed through the decoder to
# reconstruct the input at time t. The copy is run through the propagator first to yield a
# latent z_(t+dt), which is then passed through the same decoder to reconstruct the input
# at time t+dt (x_(t+dt))
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
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
import copy

# import from other modules in the folder
from VAE import VAE
from tica import TICA

from loader import nor_sim_data # normalized dimensions
from loader import raw_sim_data
from loader import pca_sim_data
from loader import tla_sim_data
from loader import con_sim_data
from loader import dt           # time lag used in dt
from loader import whiten
print("... finished loading!")

# torch.manual_seed(1)
# np.random.seed(1)


data_type = torch.float
# data_type = torch.double
threads = 4 # number of threads to use to train
model_param_fname = "data/model_parameters"

###  Network Hyperparameters  ###
variational = True
time_lagged = True
propagator  = True
denoising   = True

if time_lagged:
    sim_data = tla_sim_data
else:
    sim_data = con_sim_data

n_epochs = 100
batch_size = 100

# Network dimensions
# in_dim = 8 # input dimension
in_dim = sim_data.data[:].shape[1]
if time_lagged:
    in_dim //= 2

n_z = 1 # dimensionality of latent space

# h_size = in_dim + 2 # size of hidden layers -- don't have to all be the same size though!
h_size   = int(np.sqrt(in_dim/n_z) * n_z) + 2
h_size_0 = int((in_dim/n_z)**(2/3) * n_z) + 2
h_size_1 = int((in_dim/n_z)**(1/3) * n_z) + 2
dropout_input  = 0.5
dropout_hidden = 0.6
dropout_low    = 0.2

mean_act = nn.ReLU
var_act  = nn.ReLU

# the layers themselves
encode_layers_means = [nn.Dropout(dropout_input),
                       nn.Linear(in_dim, h_size_0),
                       nn.Dropout(dropout_hidden),
                       mean_act(),

                       nn.Linear(h_size_0, h_size_1),
                       nn.Dropout(dropout_low),
                       mean_act(),

                       # nn.Linear(h_size_1, h_size_1),
                       # nn.Dropout(dropout_low),
                       # mean_act(),

                       nn.Linear(h_size_1, n_z),
                       # just linear combination without activation
                      ]

encode_layers_vars  = [nn.Dropout(dropout_input),
                       nn.Linear(in_dim, h_size_0),
                       nn.Dropout(dropout_hidden),
                       var_act(),

                       nn.Linear(h_size_0, h_size_1),
                       nn.Dropout(dropout_low),
                       var_act(),

                       # nn.Linear(h_size_1, h_size_1),
                       # nn.Dropout(dropout_hidden),
                       # var_act(),

                       nn.Linear(h_size_1, n_z),
                      ]

decode_layers_means = [nn.Linear(n_z, h_size_1),
                       nn.Dropout(dropout_low),
                       mean_act(),

                       # nn.Linear(h_size_1, h_size_1),
                       # nn.Dropout(dropout_hidden),
                       # mean_act(),

                       nn.Linear(h_size_1, h_size_0),
                       nn.Dropout(dropout_low),
                       mean_act(),

                       nn.Linear(h_size_0, in_dim)
                      ]

decode_layers_vars  = [nn.Linear(n_z, h_size_1),
                       nn.Dropout(dropout_low),
                       var_act(),

                       nn.Linear(h_size_1, h_size_0),
                       nn.Dropout(dropout_hidden),
                       var_act(),

                       nn.Linear(h_size_0, in_dim, bias = False)
                      ]

propagator_layers   = [nn.Linear(n_z, n_z+3),
                       nn.ReLU(),
                       nn.Linear(n_z+3, n_z+3),
                       nn.ReLU(),
                       nn.Linear(n_z+3, n_z),
                      ]

if not propagator:
    propagator_layers = None
discount = 1.0

## Learning Algorithm Hyperparameters
optim_fn = optim.Adam
# optim_fn = optim.SGD
    # different optimization algorithms -- Adam tries to adaptively change momentum (memory of
    # changes from the last update) and has different learning rates for each parameter.
    # But standard Stochastic Gradient Descent is supposed to generalize better...
lr = 1e-3            # learning rate
weight_decay = 1e-6    # weight decay -- how much of a penalty to give to the magnitude of network weights (idea being that it's
    # easier to get less general results if the network weights are too big and mostly cancel each other out (but don't quite))
momentum = 1e-5        # momentum -- only does anything if SGD is selected because Adam does its own stuff with momentum.
denoise_sig = 0.001

pxz_var_init = -np.log(500) # How much weight to give to the KL-Divergence term in loss?
    # Changing this is as if we had chosen a sigma differently for (pred - truth)**2 / sigma**2, but parameterized differently.
    # See Doersch's tutorial on autoencoders, pg 14 (https://arxiv.org/pdf/1606.05908.pdf) for his comment on regularization.

# Finish preparing data
train_amt = 0.7 # fraction of samples used as training
val_amt   = 0.25 # fractions of samples used as validation
test_amt  = 1 - train_amt - val_amt

# DataLoaders to help with minibatching
# Note that num_workers runs the load in parallel
if (train_amt == 1):
    train_loader = DataLoader(sim_data, batch_size = batch_size, shuffle = True, num_workers = threads)
    val_set = None
    test_set = None
else:
    delay = 0
    tot_size = len(sim_data)
    # if time_lagged:
    #     delay = 2 * dt
    #     tot_size -= 2 * delay # wait between sets to decrease correlation

    # Scale fractions to number of examples
    train_size = int(tot_size * train_amt)
    train_size -= train_size % threads
    val_size   = int(tot_size * val_amt)
    test_size  = tot_size - (train_size + val_size)

    # Find indices
    val_start  = train_size + delay
    test_start = val_start+val_size + delay

    # Shuffle the order of examples
    rnd_sim_data = sim_data.get_permutation()

    # Make the sets and training loader
    train_set  = rnd_sim_data.get_slice(0, train_size)
    val_set    = rnd_sim_data.get_slice(val_start, val_start+val_size)
    test_set   = rnd_sim_data.get_slice(test_start, test_start+test_size)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = threads)


###  train the network!  ###
def trainer(model, optimizer, epoch, models, loss_array, kl_lambda = 1):
    """Train for one epoch

    Arguments:
        model     (VAE): the model to be run and trained
        optimizer (fxn): usually optim.Adam or optim.SGD
        epoch     (int): the number of the current epoch
        models  (list of VAEs): array to hold old models
        loss_array  (list of floats): list of rec losses
    """
    epoch_fail = 0 # loss for the epoch # credit to xkcd for the variable name

    # run through the mini batches!
    batch_count = 0 # Keep track of how many batches we've run through
    model.train() # set training mode
    for b_idx, all_data in enumerate(train_loader):
        # erase old gradient info
        optimizer.zero_grad()
        # split up the data for the time-lagged autoencoder
        # and doesn't do anything if there's no time lag
        train_data = all_data[:,:in_dim]
        goal_data  = all_data[:,-in_dim:]

        # Denoise as suggested by http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf (section 2.3 with q_D)
        # Idea is that you add noise and try to reconstruct the original (which itself has noise..)
        if denoising:
            train_data += torch.tensor(np.random.normal(0, denoise_sig, train_data.shape), dtype=data_type)

        # run the model
        fut_steps, recon_means, recon_lvs = model(train_data)

        # get loss
        if fut_steps == 0:
            loss     = model.vae_loss(recon_means[0], recon_lvs[0], goal_data, kl_lambda=kl_lambda)
            rec_loss = model.rec_loss.item()/train_data.shape[0] # reconstruction loss
        else:
            loss     =  model.vae_loss(recon_means[0], recon_lvs[0], goal_data, kl_lambda=kl_lambda)
            rec_loss =  model.rec_loss.item()/train_data.shape[0] # reconstruction loss
            loss     += model.vae_loss(recon_means[1], recon_lvs[1], goal_data, kl_lambda=kl_lambda) * model.discount
            rec_loss += model.rec_loss.item()/train_data.shape[0] # reconstruction loss
        loss_scalar = loss.item()
        epoch_fail += rec_loss

        # backprop + take a step
        loss.backward()  # Ask PyTorch to differentiate the loss w.r.t. parameters in the model
        optimizer.step() # Ask Adam/SGD to make updates to our parameters

        # print out stuff
        batch_count += 1
        if b_idx % 50 == 0:
            print("Train Epoch %d, \tBatch index %d:   \tLoss: %f\tRecLoss: %f" % (epoch, b_idx, loss_scalar, rec_loss))
            # models.append(copy.deepcopy(vae_model.state_dict()))
    print("Epoch %d has an average reconstruction loss of  %f" % (epoch, epoch_fail/batch_count))
    # different from the reconstruction loss for the most up-to-date model
    # epoch_loss = ((outputs - sim_data.data)**2).sum(1).mean() # this is pretty unstable for some reason
    models.append(copy.deepcopy(vae_model.state_dict()))
    loss_array.append(epoch_fail/batch_count) # keep track of loss

def test(model, dataset):
    """Takes a torch dataset/tensor as data and runs it through the model"""
    model.eval()
    data = torch.tensor(dataset.data[:], dtype = data_type)
    inputs  = data[..., :in_dim]
    answers = data[..., -in_dim:]
    fut_steps, out_means, out_lvs = model(inputs)
    if fut_steps == 0:
        loss = model.vae_loss(out_means[0], out_lvs[0], answers).item()
        rec_loss = model.rec_loss.item() / len(dataset)
        # rec_loss = square_loss(out_means[0], answers).item()
    else:
        loss = model.vae_loss(out_means[0], out_lvs[0], inputs)
        rec_loss = model.rec_loss.item()/len(dataset) # reconstruction loss
        # rec_loss = square_loss(out_means[0], inputs).item()

        loss += model.discount * model.vae_loss(out_means[1], out_lvs[1], answers)
        rec_loss += model.rec_loss.item()/len(dataset) # reconstruction loss
        # rec_loss = square_loss(out_means[1], answers).item()
    return loss, rec_loss

def plot_prop(range = (-3,3)):
    """Plot the output of the propagator to see """
    xs = np.arange(*range, 0.01)
    ys = vae_model.propagator_net(torch.tensor(xs, dtype=torch.float)[:, np.newaxis]).detach().numpy()
    plt.plot(xs, ys)
    plt.plot(xs, xs)
    plt.show()

square_loss = lambda y, fx: ((y-fx)**2).sum(1).mean()

### Now for the actual running and comparison against other methods ###

if __name__ == "__main__":
    # if statement is to make sure the networks can be imported to other files without this code running
    # However, if you want access to the variables by command line while running the script in interactive
    # mode, you may want to move everything outside the if statement.

    # Easily switch the data sets the comparison is using
    current_set = train_set # or sim_data

    # Compare against a naive guess of averages
    naive = current_set[:,:in_dim].mean(0)
    # naive_loss = ((naive - sim_data.data[:,-in_dim:])**2).sum(1).mean()
    naive_loss = square_loss(current_set.data[:,:in_dim], naive)
    print("A naive guess from taking averages of positions yields a loss of", naive_loss)

    # Compare against PCA
    pca_start = time.time()
    pca = PCA()
    pca_latent = pca.fit_transform(current_set.data[:,:in_dim])
    pca_latent[:,n_z:] = 0
    pca_guesses = pca.inverse_transform(pca_latent)
    # pca_loss = ((pca_guesses - sim_data.data[:,-in_dim:])**2).sum(1).mean()
    pca_loss = square_loss(current_set.data[:,-in_dim:], pca_guesses)
    print("PCA gets a reconstruction loss of %f in %f seconds" % (pca_loss, time.time()-pca_start))

    # Compare against TICA
    tica_start = time.time()
    whitened = whiten(current_set.data[:,:in_dim])
    tica_comp = TICA(whitened, dt, kinetic_map_scaling = False)
    tica_latent = tica_comp.transform(whitened)
    tica_latent[:,n_z:] = 0
    tica_guesses = tica_comp.inv_transform(tica_latent)
    # tica_loss = square_loss(sim_data.data[:,-in_dim:], tica_guesses)
    tica_loss = square_loss(current_set.data[:,-in_dim:], tica_guesses)
    print("TICA gets a reconstruction loss of %f in %f seconds" % (tica_loss, time.time()-tica_start))
    # import pdb; pdb.set_trace()

    # Compare against the desired result
    desired = np.zeros((current_set.data.shape[0], in_dim))
    desired[:,0] = current_set.data[:,0]
    for i in range(desired.shape[1]-1):
        desired[:,i+1] = current_set.data[:,i+1].mean()
    # des_loss = ((desired - sim_data.data[:,-in_dim:])**2).sum(1).mean()
    des_loss = square_loss(current_set.data[:,-in_dim:], desired)
    print("Remembering just the first axis gives a loss of", des_loss)

    # Prepare for visualization
    inputs = sim_data.data[:,:in_dim]
    H, x_edges, y_edges = np.histogram2d(inputs[:,0], inputs[:,1], bins = 30)
    x_pts = 0.5 * (x_edges[:-1]+x_edges[1:])
    y_pts = 0.5 * (y_edges[:-1]+y_edges[1:])
    grid = np.transpose([np.tile(x_pts, y_pts.shape[0]), np.repeat(y_pts, x_pts.shape[0])])
    grid = torch.tensor(grid, dtype = data_type)
    # can imshow H or run grid through vae_model.just_encode(grid)[0]

    # Initialize the model
    vae_model = VAE(in_dim              = in_dim,
                    n_z                 = n_z,
                    encode_layers_means = encode_layers_means,
                    encode_layers_vars  = encode_layers_vars,
                    decode_layers_means = decode_layers_means,
                    decode_layers_vars  = decode_layers_vars,
                    propagator_layers   = propagator_layers,
                    time_lagged         = time_lagged,
                    variational         = variational,
                    pxz_var_init        = pxz_var_init,
                    discount            = discount,
                    data_type           = data_type,
                    pref_dataset        = sim_data.data[:])

    # If you want to save/load the model's parameters
    save_model = lambda model: torch.save(model, model_param_fname)
    load_model = lambda: torch.load(model_param_fname)

    # set the learning function
    if optim_fn == optim.SGD:
        optimizer = optim_fn(vae_model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
    else:
        optimizer = optim_fn(vae_model.parameters(), lr = lr, weight_decay = weight_decay)

    # miscellaneous book-keeping variables
    models = [] # Stores old models in case an older one did better
    loss_array = []
    val_loss_array = []
    weight = 0.999 #0.9999
    wavg_loss_array = []
    start_time = time.time() # Keep track of run time
    kl_lambda = 0.1 # 1

    # Now actually do the training
    for epoch in range(n_epochs):
        kl_lambda = np.clip(epoch/10, 0.1, 1) # slowly anneal

        trainer(vae_model, optimizer, epoch, models, loss_array, kl_lambda)
        val_loss, val_rec_loss = test(vae_model, val_set)
        print("Got %f validation loss (%f reconstruction)" % (val_loss, val_rec_loss))
        # val_loss_array.append(val_loss)
        val_loss_array.append(val_rec_loss)
        train_outs = vae_model.run_data(train_set)
        true_train_mse = square_loss(train_set.data[:,-in_dim:], train_outs)

        val_outs = vae_model.run_data(val_set, future = True)
        true_val_mse = square_loss(val_set.data[:,-in_dim:], val_outs)

        wavg_loss = (1-weight) * np.array(val_loss_array[-1]) \
                    + weight   * np.array(true_val_mse)

        # wavg_loss = weight       * np.array(val_loss_array[-1]) \
        #             + (1-weight) * np.array(loss_array[-1])
        wavg_loss_array.append(wavg_loss)

        print("True Training MSE: %f" % true_train_mse)
        print("True Valid'n  MSE: %f" % true_val_mse)
        duration = time.time() - start_time
        print("%f seconds have elapsed since the training began\n" % duration)

        if epoch == 3:
            vae_model.varnet_weight += 1

        cutoff = 5
        # if epoch >= 1+cutoff and val_rec_loss > max(val_loss_array[-cutoff-1:-2]):
        if epoch >= 10 and wavg_loss > max(wavg_loss_array[-cutoff-1:-2]):
            print("Stopping training since the validation error has increased over the past %d epochs.\n"
                  "Replacing vae_model with the most recent model with lowest total validation loss." % cutoff)
            break

    # idx = len(val_loss_array) - 1 - np.argmin((val_loss_array+np.linspace(0, -0.01, len(val_loss_array)))[::-1]) # gives the most recent model with lowest validation error but biases more recent runs

    idx = len(wavg_loss_array) - 1 \
          - np.argmin(wavg_loss_array[::-1])
    _ = vae_model.load_state_dict(models[idx])
    print("Selecting the model from epoch", idx)

    # vae_model.plot_test(test_set)

    vae_model.latent_plot(mode = 'rr')
    vae_model.latent_plot(mode = 'd', axes = (0,))
    vae_model.plot_test(axes = (6,1), dims = (0,1,2,3,99))
    # vae_model.plot_test()
    plot_prop()

    if False:
        for i in range(len(models)):
            print(i)
            _ = vae_model.load_state_dict(models[i])
            vae_model.latent_plot2d('rr')
        _ = vae_model.load_state_dict(models[idx])