"""
Train an MNIST RBM, based off the demo code at
    http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
"""

import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import theano.sandbox.rng_mrg

import plotting

plt.ion()


class RBM(object):

    # --- define RBM parameters
    def __init__(self, vis_shape, n_hid,
                 input=None, W=None, c=None, b=None,
                 gaussian=None, rf_shape=None, seed=9):
        self.dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        self.gaussian = gaussian
        self.seed = seed
        self.input = input if input is not None else tt.dmatrix('input')

        rng = np.random.RandomState(seed=self.seed)
        self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=self.seed)

        # create initial weights and biases
        if W is None:
            Wmag = 4 * np.sqrt(6. / (self.n_vis + self.n_hid))
            W = rng.uniform(
                low=-Wmag, high=Wmag, size=(self.n_vis, self.n_hid)
            ).astype(self.dtype)

        if c is None:
            c = np.zeros(self.n_hid, dtype=self.dtype)

        if b is None:
            b = np.zeros(self.n_vis, dtype=self.dtype)

        # create initial sparsity mask
        self.rf_shape = rf_shape
        if rf_shape is not None:
            # assert isinstance(vis_shape, tuple) and len(vis_shape) == 2
            M, N = vis_shape
            m, n = rf_shape

            # find random positions for top-left corner of each RF
            i = rng.randint(low=0, high=M-m+1, size=self.n_hid)
            j = rng.randint(low=0, high=N-n+1, size=self.n_hid)

            mask = np.zeros((M, N, self.n_hid), dtype='bool')
            for k in xrange(self.n_hid):
                mask[i[k]:i[k]+m, j[k]:j[k]+n, k] = True

            self.mask = mask.reshape(self.n_vis, self.n_hid)
            W = W * self.mask  # make initial W sparse
        else:
            self.mask = None

        # create states for weights and biases
        W = W.astype(self.dtype)
        c = c.astype(self.dtype)
        b = b.astype(self.dtype)

        self.W = theano.shared(W, name='W')
        self.c = theano.shared(c, name='c')
        self.b = theano.shared(b, name='b')

        # create states for initial increments (for momentum)
        self.Winc = theano.shared(np.zeros_like(W), name='Winc')
        self.cinc = theano.shared(np.zeros_like(c), name='cinc')
        self.binc = theano.shared(np.zeros_like(b), name='binc')

    @property
    def filters(self):
        if self.mask is None:
            return self.W.get_value().T.reshape((self.n_hid,) + self.vis_shape)
        else:
            filters = self.W.get_value().T[self.mask.T]
            shape = (self.n_hid,) + self.rf_shape
            return filters.reshape(shape)

    # --- define RBM propagation functions
    def probHgivenV(self, vis):
        hidprob = tt.nnet.sigmoid(tt.dot(vis, self.W) + self.c)
        return hidprob

    def probVgivenH(self, hid):
        a = tt.dot(hid, self.W.T) + self.b
        if self.gaussian is None:
            visprob = tt.nnet.sigmoid(a)
        else:
            visprob = self.gaussian * a

        return visprob

    def sampHgivenV(self, vis):
        hidprob = self.probHgivenV(vis)
        hidsamp = self.theano_rng.binomial(
            size=hidprob.shape, n=1, p=hidprob, dtype=self.dtype)
        return hidprob, hidsamp

    # --- define RBM updates
    def get_cost_updates(self, rate=0.05, weightcost=2e-4, momentum=0.5):

        numcases = self.input.shape[0]
        rate = tt.cast(rate, self.dtype)
        weightcost = tt.cast(weightcost, self.dtype)
        momentum = tt.cast(momentum, self.dtype)

        # compute positive phase
        data = self.input
        poshidprob, poshidsamp = self.sampHgivenV(data)

        posprods = tt.dot(data.T, poshidprob) / numcases
        posvisact = tt.mean(data, axis=0)
        poshidact = tt.mean(poshidprob, axis=0)

        # compute negative phase
        negdata = self.probVgivenH(poshidsamp)
        neghidprob = self.probHgivenV(negdata)
        negprods = tt.dot(negdata.T, neghidprob) / numcases
        negvisact = tt.mean(negdata, axis=0)
        neghidact = tt.mean(neghidprob, axis=0)

        # compute error
        rmse = tt.sqrt(tt.mean((data - negdata)**2, axis=1))
        err = tt.mean(rmse)

        # compute updates
        Winc = momentum * self.Winc + rate * (
            (posprods - negprods) - weightcost * self.W)
        cinc = momentum * self.cinc + rate * (poshidact - neghidact)
        binc = momentum * self.binc + rate * (posvisact - negvisact)

        # if self.mask is not None:
        #     Winc = Winc * self.mask

        updates = [
            (self.W, self.W + Winc),
            (self.c, self.c + cinc),
            (self.b, self.b + binc),
            (self.Winc, Winc),
            (self.cinc, cinc),
            (self.binc, binc)
        ]

        return err, updates

# --- load the data
# def show(flat_image, ax=None):
#     ax = ax or plt.gca()
#     ax.imshow(flat_image.reshape(28, 28), cmap='gray', interpolation='none')

filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)


if 0:
    # make each pixel zero mean and unit std
    for images, labels in [train, valid, test]:
        images -= images.mean(axis=0, keepdims=True)
        images /= np.maximum(images.std(axis=0, keepdims=True), 1e-3)


if 1:
    plt.figure(1)
    plt.clf()
    # print train[0][10]
    plotting.show(train[0][10].reshape(28, 28))


# --- train
images, labels = train
n_epochs = 10
n_vis = images.shape[1]
n_hid = 500

batch_size = 100
batches = images.reshape(
    images.shape[0] / batch_size, batch_size, images.shape[1])

# rbm = RBM(n_vis, n_hid)
# rbm = RBM((28, 28), n_hid)
rbm = RBM((28, 28), n_hid, rf_shape=(9, 9))
cost, updates = rbm.get_cost_updates()

train_rbm = theano.function([rbm.input], cost,
                            updates=updates)


hp = rbm.probHgivenV(rbm.input)
vp = rbm.probVgivenH(hp)
reconstruct_rbm = theano.function([rbm.input], vp)

reconstruct_rbm(batches[0])

for epoch in range(n_epochs):

    costs = []
    for batch in batches:
        costs.append(train_rbm(batch))

    print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

    # weights = rbm.W.get_value()
    plt.figure(2)
    plt.clf()
    # plotting.filters(weights.T.reshape(-1, 28, 28), rows=5, cols=10)
    plotting.filters(rbm.filters, rows=10, cols=20)

    test_batch = batches[0]
    recons = reconstruct_rbm(test_batch)
    plt.figure(3)
    plt.clf()
    plotting.compare([test_batch.reshape(-1, 28, 28), recons.reshape(-1, 28, 28)], rows=5, cols=20)

    plt.draw()
