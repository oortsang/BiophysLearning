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

# Get the MNIST data
import torchvision
from torchvision import datasets


###  Initializing data set  ###

from loader import sim_data
# from loader import raw_sim_data as sim_data

# # vxlabs' guide says transform = torchvision.transforms.ToTensor()
# mnist_trainset = datasets.MNIST(root='./data', train=True,  download=True, transform=torchvision.transforms.ToTensor())
# mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

print("... finished l0ad1ng!")

# torch.random.manual_seed(17)

###  A couple Hyperparameters  ###
n_epochs = 20
bsize = 20 # batch size
# smaller batches seem to be able to get out of local minima better

print("Batch size:", bsize)

train_loader = DataLoader(sim_data, batch_size = bsize, shuffle = True, num_workers = 4)
test_loader = DataLoader(sim_data, batch_size = 400, shuffle = False, num_workers = 4)

# train_loader = DataLoader(mnist_trainset, batch_size = bsize, shuffle=True, num_workers=4)
# test_loader  = DataLoader(mnist_testset,  batch_size = bsize, shuffle=True)
data_type = torch.float


in_dim  = 16

hsize1_enc = 4
n_z = 2 # dimensionality of latent space

hsize1_dec = hsize1_enc

lr = 1e-4
weight_decay = 0
pmomentum = 1e-2

# holds both encoder and decoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.always_random_sample = False # for the sampling process of latent variables

        # set up the encoder
        # one for means, one for variance -- there's no reason they should share layers
        self.encode_layers_means = [nn.Linear(in_dim, hsize1_enc),
                                    nn.ReLU(),
                                    nn.Linear(hsize1_enc, n_z),
                                   ]
        self.encode_layers_vars = [nn.Linear(in_dim, hsize1_enc),
                                   nn.ReLU(),
                                   nn.Linear(hsize1_enc, n_z)
                                  ]
        self.encode_net_means = nn.Sequential(*self.encode_layers_means)
        self.encode_net_vars  = nn.Sequential(*self.encode_layers_vars)


        # set up the decoder
        # only take in n_z because we sample from the probability distribution
        self.decode_layers = [nn.Linear(n_z, hsize1_dec),
                              nn.ReLU(),
                              nn.Linear(hsize1_dec, in_dim)# ,
                              # nn.Sigmoid()
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
        self.mu     = self.encode_net_means(x)
        self.logsig = self.encode_net_vars(x)
        self.z = (self.mu, self.logsig)
        return self.z

    # Does the sampling outside
    def sample(self, mu, logsig):
        # In training mode, we want to carry the propagation through
        # otherwise, we just return the maximum likelihood mu
        # Can always change this in the __init__ function
        res = mu
        if self.training or self.always_random_sample:
            epsshape = (bsize, n_z) if logsig.dim == 2 else n_z
            eps = torch.tensor(np.random.normal(0, 1, epsshape), dtype=data_type)
            # a randomly pulled number for each example in the minibatch
            res += torch.exp(logsig*0.5) * eps # elementwise multiplication is desired
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
        # Decoding seems to be returning the max-likelihood output rathe rthan a full prob dist
        return x_decoded

    # # Use dimension-wise square loss since it's Gaussian...?
    def vae_loss(self, pred, truth):
        # ***** may need to reshape pred to help the comparison
        # pred.reshape(truth.shape) # but might not be necessary
        # E[log P (x | z)]
        # rec_loss = (nn.BCELoss(reduction = 'sum'))(pred, truth) # doesn't work for some reason...
        rec_loss = torch.sum((pred - truth)**2)

        # KL[Q(z|x) || P(z|x)]
        # as appears in https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
        # but probably in some paper (2-3 lines from Doersch + diagonal covariance matrix assn.)
        # kl_div   = 0.5 * torch.sum(torch.exp(self.logsig)+self.mu*self.mu-1-self.logsig, dim=1)
        kl_div   = 0.5 * torch.sum(torch.exp(self.logsig)+self.mu*self.mu-1-self.logsig)
        # also maybe try to replace with a pytorch library fxn later...
        return (rec_loss + kl_div)/bsize

###  train the network!  ###
vae_model = VAE()
# optimizer = optim.Adam(vae_model.parameters(), lr = lr, weight_decay = weight_decay)
optimizer = optim.SGD(vae_model.parameters(), lr = lr, weight_decay = weight_decay, momentum = pmomentum)
loss_array = []
models = []
# run one epoch
def trainer(model, epoch):
    epoch_fail = 0 # loss for the epoch
    # run through the mini batches!
    bcount = 0
    for b_idx, data in enumerate(train_loader):
        # if b_idx == 45:
        #     pdb.set_trace()
        model.train() # set training mode
        optimizer.zero_grad()
        # data = data.view(-1, in_dim) # reshape to make things play nicely
        recon = model(data)
        loss  = model.vae_loss(recon, data)
        loss_scalar = loss.item()
        epoch_fail += loss_scalar
        loss_array.append(loss_scalar)
        loss.backward()
        optimizer.step()
        bcount += 1
        if b_idx % 100 == 0:
            print("Train Epoch %d, \tBatch index %d:   \tLoss: %f" % (epoch, b_idx, loss_scalar))
    print("Epoch %d has average loss of %f" % (epoch, epoch_fail/bcount))
    models.append(vae_model)

# try to recreate the original data set
def test():
    vae_model.eval()
    outputs = np.zeros(sim_data.data.shape, dtype=np.float32)
    for b_idx, data in enumerate(test_loader):
        recon = vae_model(data).detach().numpy()
        csize = data.shape[0]
        outputs[b_idx*bsize:b_idx*bsize+csize, :] = recon
    return outputs

naive = sim_data[-40000:].mean(0)
naive_loss = ((naive - sim_data.data)**2).sum(1).mean()

print("A naive guess from taking averages of the last 40000 positions yields a loss of", naive_loss)


# Now actually do the training
start_time = time.time()
for epoch in range(n_epochs):
    trainer(vae_model, epoch)
    duration = time.time() - start_time
    print("%f seconds have elapsed since the training began\n" % duration)
    #validate(epoch) # to tell you how it's doing on the test set


        
