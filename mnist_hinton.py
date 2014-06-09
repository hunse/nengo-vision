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
                 W=None, c=None, b=None, mask=None,
                 gaussian=None, rf_shape=None, seed=9):
        self.dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        self.gaussian = gaussian
        self.seed = seed
        self.pre = None
        self.post = None

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
        self.mask = mask
        if rf_shape is not None and mask is None:
            assert isinstance(vis_shape, tuple) and len(vis_shape) == 2
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

    def save(self, filename):
        d = dict()
        for k, v in self.__dict__.items():
            if k in ['W', 'c', 'b']:
                d[k] = v.get_value()
            elif k in ['vis_shape', 'n_hid', 'rf_shape', 'mask', 'seed']:
                d[k] = v
        np.savez(filename, dict=d)

    @classmethod
    def load(cls, filename):
        d = np.load(filename)['dict'].item()
        return cls(**d)

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
    def get_cost_updates(self, data, rate=0.05, weightcost=2e-4, momentum=0.5):

        numcases = data.shape[0]
        rate = tt.cast(rate, self.dtype)
        weightcost = tt.cast(weightcost, self.dtype)
        momentum = tt.cast(momentum, self.dtype)

        # compute positive phase
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

        if self.mask is not None:
            Winc = Winc * self.mask

        updates = [
            (self.W, self.W + Winc),
            (self.c, self.c + cinc),
            (self.b, self.b + binc),
            (self.Winc, Winc),
            (self.cinc, cinc),
            (self.binc, binc)
        ]

        return err, updates

    @property
    def encode(self):
        data = tt.dmatrix('data')
        code = self.probHgivenV(data)
        return theano.function([data], code)

    def get_output(self, images):
        data = images if self.pre is None else self.pre.get_output(images)
        return self.probHgivenV(data)

    def get_reconstruction(self, code):
        data = self.probVgivenH(code)
        return (data if self.pre is None else
                self.pre.get_reconstruction(data))

    @property
    def reconstruct(self):
        assert self.post is None
        images = tt.dmatrix('images')
        output = self.get_output(images)
        recons = self.get_reconstruction(output)
        return theano.function([images], recons)

    def pretrain(self, batches, test_images, n_epochs=10):

        data = tt.dmatrix('data')
        cost, updates = self.get_cost_updates(data)
        train_rbm = theano.function(
            [data], cost, updates=updates)

        for epoch in range(n_epochs):

            costs = []
            for batch in batches:
                costs.append(train_rbm(batch))

            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

            plt.figure(2)
            plt.clf()
            recons = self.reconstruct(test_images)
            plotting.compare([test_images.reshape(-1, 28, 28),
                              recons.reshape(-1, 28, 28)],
                             rows=5, cols=20)

            if self.pre is None:
                # plot filters for first layer only
                plt.figure(3)
                plt.clf()
                plotting.filters(self.filters, rows=10, cols=20)

            plt.draw()


def link_rbms(pre, post):
    pre.post = post
    post.pre = pre


# --- load the data
filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)

# --- train
shapes = [(28, 28), 500, 200, 50]
rf_shapes = [(9, 9), None, None]
assert len(shapes) == len(rf_shapes) + 1

images, labels = train
test_images = images[:100]
n_epochs = 2
batch_size = 100

data = images
rbms = []
for i in range(len(rf_shapes)):
    savename = "layer%d.npz" % i
    if not os.path.exists(savename):
        batches = data.reshape(
            data.shape[0] / batch_size, batch_size, data.shape[1])

        rbm = RBM(shapes[i], shapes[i+1], rf_shape=rf_shapes[i])
        if len(rbms) > 0:
            link_rbms(rbms[-1], rbm)

        rbm.pretrain(batches, test_images, n_epochs=n_epochs)
        rbm.save(savename)
    else:
        rbm = RBM.load(savename)
        if len(rbms) > 0:
            link_rbms(rbms[-1], rbm)

    data = rbm.encode(data)
    rbms.append(rbm)


plt.figure(99)
plt.clf()
recons = rbm.reconstruct(test_images)
plotting.compare([test_images.reshape(-1, 28, 28),
                  recons.reshape(-1, 28, 28)],
                 rows=5, cols=20)
