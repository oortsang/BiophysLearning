#!/usr/bin/python
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

from loader import sim_data as sim_data
# from loader import raw_sim_data as sim_data
# from loader import pca_sim_data as sim_data

print("... finished loading!")

###  A couple Hyperparameters  ###
n_epochs = 20
batch_size = 80 # batch size
# smaller batches seem to be able to get out of local minima better
# but larger batches provide better stability
print("Batch size:", batch_size)


train_loader = DataLoader(sim_data, batch_size = batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(sim_data, batch_size = 400, shuffle = False, num_workers = 4)

data_type = torch.float # helps with consistency of type -- alternative is torch.double

# Network dimensions
in_dim  = 5 # input dimension
hsize = 4   # size of hidden layers
n_z = 3  # dimensionality of latent space

# learning algorithm hyperparameters
# (Currently set to Adam rather than SGD)
lr = 1e-3         # learning rate
weight_decay = 0  # weight decay
pmomentum = 1e-4  # momentum

kl_lambda = 1 # effectively gives more weight to reconstruction loss
# as if we had chosen a sigma differently for (pred - truth)**2 / sigma**2
# See Doersch's tutorial on autoencoders, pg 14 (https://arxiv.org/pdf/1606.05908.pdf)

# holds both encoder and decoder
class VAE(nn.Module):
    def __init__(self):
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

        if data_type == torch.float:
            self.encode_net_means = self.encode_net_means.float()
            self.encode_net_vars  = self.encode_net_vars.float()
            self.decode_net       = self.decode_net.float()
        else:
            self.encode_net_means = self.encode_net_means.double()
            self.encode_net_vars  = self.encode_net_vars.double()
            self.decode_net       = self.decode_net.double()

    # Encoding from x to z
    # returns the probability distribution in terms of mean and log-var of a gaussian
    def encode(self, x):
        self.x = x
        self.mu     = self.encode_net_means(x)
        self.logvar = self.encode_net_vars(x)
        self.z = (self.mu, self.logvar)
        return self.z

    # Does the sampling outside of the backpropagation
    def sample(self, mu, logvar):
        # In training mode, we want to carry the propagation through
        # otherwise, we just return the maximum likelihood mu
        # Can always change this in the __init__ function
        res = mu
        variational = True # False basically sets it to autoencoder status (but not quite b/c of the loss function)
        if variational and self.training or self.always_random_sample:
            epsshape = (batch_size, n_z) if logvar.dim == 2 else n_z
            eps = torch.tensor(np.random.normal(0, 1, epsshape), dtype=data_type)
            # a randomly pulled number for each example in the minibatch
            res += torch.exp(logvar*0.5) * eps # elementwise multiplication is desired
        return res

    # Takes a single z sample and attempts to reconstruct a ~16-dim simulation ~
    def decode(self, z):
        if False and not self.training:
            # save the parameters... for visualization purposes
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
        z_dist    = self.encode(x)        # 1. Encode
        z_sample  = self.sample(*z_dist)  # 2. Sample
        x_decoded = self.decode(z_sample) # 3. Decode
        # Decoding seems to be returning the max-likelihood output rather than a full prob dist
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
        # kl_div   = 0.5 * torch.sum(torch.exp(self.logvar)+self.mu*self.mu-1-self.logvar, dim=1)
        kl_div   = 0.5 * torch.sum(torch.exp(self.logvar)+self.mu*self.mu-1-self.logvar)
        return (self.rec_loss + kl_lambda*kl_div)/batch_size

    def plot_last_run(self, axes = (3,2)):
        # want to trace how the reconstruction compares to the input
        input = self.x.detach().numpy()
        saved = self.z[0].detach().numpy()
        output = self.forward(self.x).detach().numpy()
        plt.subplot(*axes, 1)
        plt.title("Latent variables")
        plt.plot(saved)
        
        for part in range(input.shape[1]):
            plt.subplot(*axes,2+part)
            plt.title("Particle %d" % part+1)
            plt.plot(input[:,part], label = "Input")
            plt.plot(output[:,part], label = "Reconstructed guess")
            plt.legend()
        plt.show()

###  train the network!  ###
vae_model = VAE()
# optimizer = optim.Adam(vae_model.parameters(), lr = lr, weight_decay = weight_decay)
optimizer = optim.SGD(vae_model.parameters(), lr = lr, weight_decay = weight_decay, momentum = pmomentum)
# loss_array = []
models = []
# run one epoch
def trainer(model, epoch):
    epoch_fail = 0 # loss for the epoch
    # run through the mini batches!
    bcount = 0
    for b_idx, data in enumerate(train_loader):
        model.train() # set training mode
        optimizer.zero_grad()
        recon = model(data)
        loss  = model.vae_loss(recon, data)
        loss_scalar = loss.item()
        rec_loss = model.rec_loss.item()/batch_size
        epoch_fail += rec_loss
        loss.backward()
        optimizer.step()
        bcount += 1
        if b_idx % 100 == 0:
            print("Train Epoch %d, \tBatch index %d:   \tLoss: %f\tRecLoss: %f" % (epoch, b_idx, loss_scalar, rec_loss))
    # outputs = model(torch.tensor(sim_data.data[:])).detach().numpy()
    # epoch_loss = ((outputs - sim_data.data)**2).sum(1).mean()
    print("Epoch %d has an average reconstruction loss of  %f" % (epoch, epoch_fail/bcount))
    # not accurate though because this isn't looking at the reconstruction loss at the end of the epoch...
    # but recomputing everything gives a avery unstable output
    models.append(vae_model)

# try to recreate the original data set
def test():
    vae_model.eval()
    outputs = np.zeros(sim_data.data.shape, dtype=np.float32)
    for b_idx, data in enumerate(test_loader):
        recon = vae_model(data).detach().numpy()
        csize = data.shape[0]
        outputs[b_idx*batch_size:b_idx*batch_size+csize,:] = recon
    return outputs

def test2(model = vae_model, plot = True, axes=(3,3), ret = False):
    bigdata = torch.tensor(sim_data.data[:])
    outputs = model(bigdata).detach().numpy()
    bigdata = bigdata.detach().numpy()
    latents = model.z[0].detach().numpy() # just give the means...
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

naive = sim_data[-40000:].mean(0)
naive_loss = ((naive - sim_data.data[:])**2).sum(1).mean()
print("A naive guess from taking averages of the last 40000 positions yields a loss of", naive_loss)

pca_start = time.time()
pca = PCA()
pca_latent = pca.fit_transform(sim_data.data[:])
pca_latent[:,n_z:] = 0
pca_guesses = pca.inverse_transform(pca_latent)
pca_loss = ((pca_guesses - sim_data.data[:])**2).sum(1).mean()
print("PCA gets a loss of %f in %f seconds" % (pca_loss, time.time()-pca_start))

# Now actually do the training
start_time = time.time()
for epoch in range(n_epochs):
    trainer(vae_model, epoch)
    duration = time.time() - start_time
    print("%f seconds have elapsed since the training began\n" % duration)
    #validate(epoch) # to tell you how it's doing on the test set
