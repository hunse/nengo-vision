"""
Based off the tutorial at http://deeplearning.net/tutorial/rbm.html
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

    def __init__(self, n_vis, n_hid, input=None, gaussian=None, seed=9):
        # --- define RBM parameters
        self.dtype = theano.config.floatX

        self.n_vis = n_vis
        self.n_hid = n_hid
        self.gaussian = gaussian
        self.seed = seed
        self.input = input if input is not None else tt.dmatrix('input')

        rng = np.random.RandomState(seed=seed)
        self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=seed)

        W0mag = 4 * np.sqrt(6. / (n_vis + n_hid))
        W0 = np.random.uniform(low=-W0mag, high=W0mag, size=(n_vis, n_hid)
                               ).astype(self.dtype)
        self.W = theano.shared(W0, name='W')
        self.c = theano.shared(np.zeros(n_hid, dtype=self.dtype), name='c')
        self.b = theano.shared(np.zeros(n_vis, dtype=self.dtype), name='b')
        self.params = [self.W, self.c, self.b]

    # --- define RBM propagation functions
    def propup(self, vis):
        a = tt.dot(vis, self.W) + self.c
        p = tt.nnet.sigmoid(a)
        return a, p

    def propdown(self, hid):
        a = tt.dot(hid, self.W.T) + self.b
        if self.gaussian is None:
            p = tt.nnet.sigmoid(a)
        else:
            p = self.gaussian * a

        return a, p

    def sampleHgivenV(self, vis):
        ha, hp = self.propup(vis)
        sample = self.theano_rng.binomial(size=hp.shape, n=1, p=hp, dtype=self.dtype)
        return ha, hp, sample

    def sampleVgivenH(self, hid):
        va, vp = self.propdown(hid)
        if self.gaussian is None:
            sample = self.theano_rng.binomial(size=vp.shape, n=1, p=vp, dtype=self.dtype)
        else:
            # TODO
            sample = vp

        return va, vp, sample

    def gibbsHVH(self, hs0):
        va1, vp1, vs1 = self.sampleVgivenH(hs0)
        ha1, hp1, hs1 = self.sampleHgivenV(vs1)
        return va1, vp1, vs1, ha1, hp1, hs1

    def gibbsVHV(self, vs0):
        ha1, hp1, hs1 = self.sampleHgivenV(vs0)
        va1, vp1, vs1 = self.sampleVgivenH(hs1)
        return ha1, hp1, hs1, va1, vp1, vs1

    def free_energy(self, v):
        ha = tt.dot(v, self.W) + self.c
        vis_term = tt.dot(v, self.b)
        hid_term = tt.sum(tt.log(1 + tt.exp(ha)), axis=1)
        return -hid_term - vis_term

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):

        # compute positive phase
        ha, hp, hs = self.sampleHgivenV(self.input)

        if persistent is None:
            chain_start = hs
        else:
            chain_start = persistent

        # perform gibbs sampling
        [vaN, vpN, vsN, haN, hpN, hsN], updates = theano.scan(
            self.gibbsHVH,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k)

        chain_end = vsN[-1]

        cost = tt.mean(self.free_energy(self.input)) - tt.mean(self.free_energy(chain_end))
        gparams = tt.grad(cost, self.params, consider_constant=[chain_end])

        for param, grad in zip(self.params, gparams):
            updates[param] = param - grad * tt.cast(lr, self.dtype)

        if persistent is not None:
            updates[persistent] = hsN[-1]

        return cost, updates

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

rbm = RBM(n_vis, n_hid)
persistent = theano.shared(np.zeros((batch_size, n_hid), dtype=rbm.dtype),
                           name='persistent')
cost, updates = rbm.get_cost_updates(lr=0.02, persistent=persistent, k=1)

train_rbm = theano.function([rbm.input], cost,
                            updates=updates)


ha, hp = rbm.propup(rbm.input)
va, vp = rbm.propdown(hp)
reconstruct_rbm = theano.function([rbm.input], vp)

for epoch in range(n_epochs):

    costs = []
    for batch in batches:
        costs.append(train_rbm(batch))

    print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

    weights = rbm.W.get_value()
    plt.figure(2)
    plt.clf()
    plotting.filters(weights.T.reshape(-1, 28, 28), rows=5, cols=10)

    test_batch = batches[0]
    recons = reconstruct_rbm(test_batch)
    plt.figure(3)
    plt.clf()
    plotting.compare([test_batch.reshape(-1, 28, 28), recons.reshape(-1, 28, 28)], rows=5, cols=20)

    plt.draw()
