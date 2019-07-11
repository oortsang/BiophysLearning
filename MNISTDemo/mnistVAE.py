print("Loading...")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb

# Get the MNIST data
import torchvision
from torchvision import datasets

###  Initializing data set  ###

# vxlabs' guide says transform = torchvision.transforms.ToTensor()
mnist_trainset = datasets.MNIST(root='./data', train=True,  download=True, transform=torchvision.transforms.ToTensor())
mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
mnist_dim  = 784 # 28 * 28

print("... finished l0ad1ng!")

# torch.random.manual_seed(17)

###  A couple Hyperparameters  ###
n_epochs = 5
bsize = 60 # batch size
train_loader = DataLoader(mnist_trainset, batch_size = bsize, shuffle=True, num_workers=4)
test_loader  = DataLoader(mnist_testset,  batch_size = bsize, shuffle=True)
data_type = torch.float

hsize1_enc = 384
hsize2_enc = 32
n_z = 2 # dimensionality of latent space

lr = 1.5e-3
weight_decay = 0

hsize1_dec = hsize2_enc
hsize2_dec = hsize1_enc

# holds both encoder and decoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.always_random_sample = False # for the sampling process of latent variables

        # set up the encoder
        self.encode_layers_means = [nn.Linear(mnist_dim, hsize1_enc),
                                    nn.ReLU(),
                                    nn.Linear(hsize1_enc, hsize2_enc),
                                    nn.ReLU(),
                                    nn.Linear(hsize2_enc, n_z)
                                   ]
        self.encode_layers_vars = [nn.Linear(mnist_dim, hsize1_enc),
                                    nn.ReLU(),
                                    nn.Linear(hsize1_enc, hsize2_enc),
                                    nn.ReLU(),
                                    nn.Linear(hsize2_enc, n_z)
                                   ]
        self.encode_net_means = nn.Sequential(*self.encode_layers_means)
        self.encode_net_vars  = nn.Sequential(*self.encode_layers_vars)


        # set up the decoder
        self.decode_layers = [nn.Linear(n_z, hsize1_dec), # only take n_z because we sample from the probability distribution
                              nn.ReLU(),
                              nn.Linear(hsize1_dec, hsize2_dec),
                              nn.ReLU(),
                              nn.Linear(hsize2_dec, mnist_dim),
                              nn.Sigmoid()
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
        # x is coming in as a 4D tensor: [50, 1, 28, 28]
        # need to figure out how to treat all of it at the same time
        # mu_and_sig = self.encode_net(x)
        # self.mu, self.logsig = mu_and_sig[:n_z], mu_and_sig[n_z:]
        self.mu     = self.encode_net_means(x)
        self.logsig = self.encode_net_vars(x)
        self.z = self.mu, self.logsig
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

    # Takes a single z sample and attempts to reconstruct a 28x28 image
    # as a 784-entry-tensor, of course
    def decode(self, z):
        if False and not self.training:
            # save the parameters... for visualization purposes
            self.post_first_linear  = self.decode_layers[0](z)
            self.activated          = self.decode_layers[1](self.post_first_linear)
            self.post_second_linear = self.decode_layers[2](self.activated)
            self.second_activated   = self.decode_layers[3](self.post_second_linear)
            self.post_third_linear  = self.decode_layers[4](self.second_activated)
            self.final_output       = self.decode_layers[5](self.post_third_linear)
            return self.final_output
        else:
            output = self.decode_net(z)
            return output

    # run through encoder, sampler, and decoder
    def forward(self, x):
        # 1. Encode
        # 2. Sample
        # 3. Decode and return the probability distribution (?)
        #    (or else return the maximum likelihood reconstruction)
        z_dist    = self.encode(x)
        z_sample  = self.sample(*z_dist)
        x_decoded = self.decode(z_sample)
        return x_decoded # output is a single value rather than probability dist...?

    # # Use pixel-wise square loss since it's Gaussian...?
    # Actually, try cross entropy loss first because both tutorials say to
    def vae_loss(self, pred, truth):
        # ***** may need to reshape pred to help the comparison
        # pred.reshape(truth.shape) # but might not be necessary
        # E[log P (x | z)]
        rec_loss = (nn.BCELoss(reduction = 'sum'))(pred, truth)
        # rec_loss = torch.sum( (pred - truth)**2)

        # KL[Q(z|x) || P(z|x)]
        # as appears in https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
        # but probably in some paper (2-3 lines from Doersch + diagonal covariance matrix assn.)
        # kl_div   = 0.5 * torch.sum(torch.exp(self.logsig)+self.mu*self.mu-1-self.logsig, dim=1)
        kl_div   = 0.5 * torch.sum(torch.exp(self.logsig)+self.mu*self.mu-1-self.logsig)
        # also maybe try to replace with a pytorch library fxn later...
        # where does logsig come from? is this really self.logsig?
        # ditto for mu and self.mu
        return (rec_loss + kl_div)/bsize

###  train the network!  ###
vae_model = VAE()
optimizer = optim.Adam(vae_model.parameters(), lr = lr, weight_decay = weight_decay)

# run one epoch
def trainer(model, epoch):
    epoch_fail = 0 # loss for the epoch
    # run through the mini batches!
    bcount = 0
    for b_idx, (data, no_peaking) in enumerate(train_loader):
        model.train() # set training mode
        optimizer.zero_grad()
        data = data.view(-1, mnist_dim) # reshape to make things play nicely
        recon = model(data)
        loss  = model.vae_loss(recon, data)
        loss_scalar = loss.item()
        epoch_fail += loss_scalar
        loss.backward()
        optimizer.step()
        bcount += 1
        if b_idx % 100 == 0:
            print("Train Epoch %d, \tBatch index %d:   \tLoss: %f" % (epoch, b_idx, loss_scalar))
    print("Epoch %d has average loss of %f\n" % (epoch, epoch_fail/bcount))

def test(epoch):
    model.eval() # eval mode
    test_loss = 0
    # loop through batches
    # turn off differentiation to go fast
    # just to visualize it...
    # can do this later
    pass

def run_model(za,zb, model=vae_model):
    vispic = vae_model.decode(torch.tensor((za,zb), dtype=data_type))
    vispic = vispic.detach().numpy().reshape((28,28))
    return vispic

def visr(bound=4, freq=0.25):
    sqlen = 28
    aarange = np.arange(-bound, bound, freq)
    bbrange = np.arange(-bound, bound, freq)
    visquilt = np.zeros((sqlen*aarange.shape[0], sqlen*bbrange.shape[0]))
    vae_model.eval()
    for i, za in enumerate(aarange):
      for j, zb in enumerate(bbrange):
        vispic = vae_model.decode(torch.tensor((za,zb), dtype=data_type))
        vispic = vispic.detach().numpy().reshape(28,28)
        visquilt[sqlen*i:sqlen*(i+1), sqlen*j:sqlen*(j+1)] = vispic
    plt.imshow(visquilt, cmap='gray')
    plt.show()
    return visquilt

def run_one_ex(i, returnval=False):
    vae_model.eval()
    orig = mnist_trainset[i][0][0]
    label = mnist_trainset[i][1]
    rec = vae_model.forward(orig.reshape(784)).reshape(28,28).detach().numpy()
    orig = orig.detach().numpy()
    comparison = np.zeros((2*28,28))
    comparison[:28,:] = orig
    comparison[28:,:] = rec
    print("Displaying example %d, which is digit %d" % (i, label))
    plt.imshow(comparison, cmap='gray')
    plt.show()
    if returnval:
        return comparison

# Now actually do the training
for epoch in range(n_epochs):
    trainer(vae_model, epoch)
    #test(epoch) # to tell you how it's doing on the test set
