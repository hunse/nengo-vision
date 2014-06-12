"""
Train an MNIST RBM, based off the demo code at
    http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
"""

import collections
import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt

# os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
import theano
import theano.tensor as tt
import theano.sandbox.rng_mrg

import plotting

plt.ion()


def norm(x, **kwargs):
    return np.sqrt((x**2).sum(**kwargs))


class RBM(object):

    # --- define RBM parameters
    def __init__(self, vis_shape, n_hid,
                 W=None, c=None, b=None, mask=None,
                 rf_shape=None, hidlinear=False, seed=22):
        self.dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        # self.gaussian = gaussian
        self.hidlinear = hidlinear
        self.seed = seed

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
        x = tt.dot(vis, self.W) + self.c
        if self.hidlinear:
            return x
        else:
            return tt.nnet.sigmoid(x)

    def probVgivenH(self, hid):
        a = tt.dot(hid, self.W.T) + self.b
        # if self.gaussian is None:
        visprob = tt.nnet.sigmoid(a)
        # else:
        #     visprob = self.gaussian * a

        return visprob

    def sampHgivenV(self, vis):
        hidprob = self.probHgivenV(vis)
        if self.hidlinear:
            hidsamp = hidprob + self.theano_rng.normal(
                size=hidprob.shape, dtype=self.dtype)
        else:
            hidsamp = self.theano_rng.binomial(
                size=hidprob.shape, n=1, p=hidprob, dtype=self.dtype)
        return hidprob, hidsamp

    # --- define RBM updates
    def get_cost_updates(self, data, rate=0.1, weightcost=2e-4, momentum=0.5):

        numcases = tt.cast(data.shape[0], self.dtype)
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
        data = tt.matrix('data', dtype=self.dtype)
        code = self.probHgivenV(data)
        return theano.function([data], code)

    def pretrain(self, batches, dbn=None, test_images=None,
                 n_epochs=10, **train_params):

        data = tt.matrix('data', dtype=self.dtype)
        cost, updates = self.get_cost_updates(data, **train_params)
        train_rbm = theano.function([data], cost, updates=updates)

        for epoch in range(n_epochs):

            # train on each mini-batch
            costs = []
            for batch in batches:
                costs.append(train_rbm(batch))

            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

            if dbn is not None and test_images is not None:
                # plot reconstructions on test set
                plt.figure(2)
                plt.clf()
                recons = dbn.reconstruct(test_images)
                plotting.compare([test_images.reshape(-1, 28, 28),
                                  recons.reshape(-1, 28, 28)],
                                 rows=5, cols=20)
                plt.draw()

            # plot filters for first layer only
            if dbn is not None and self is dbn.rbms[0]:
                plt.figure(3)
                plt.clf()
                plotting.filters(self.filters, rows=10, cols=20)
                plt.draw()


class DBN(object):

    def __init__(self, rbms=None):
        self.dtype = theano.config.floatX
        self.rbms = rbms if rbms is not None else []
        self.classifier = None

    # def encode(self, images):
    #     data = images if self.pre is None else self.pre.encode(images)
    #     return self.probHgivenV(data)

    # def decode(self, code):
    #     data = self.probVgivenH(code)
    #     return data if self.pre is None else self.pre.decode(data)

    def propup(self, images):
        codes = images
        for rbm in self.rbms:
            codes = rbm.probHgivenV(codes)
        return codes

    def propdown(self, codes):
        images = codes
        for rbm in self.rbms[::-1]:
            images = rbm.probVgivenH(images)
        return images

    @property
    def encode(self):
        images = tt.matrix('images', dtype=self.dtype)
        codes = self.propup(images)
        return theano.function([images], codes)

    @property
    def decode(self):
        codes = tt.matrix('codes', dtype=self.dtype)
        images = self.propdown(codes)
        return theano.function([codes], images)

    @property
    def reconstruct(self):
        images = tt.matrix('images', dtype=self.dtype)
        codes = self.propup(images)
        recons = self.propdown(codes)
        return theano.function([images], recons)

    def get_categories_vocab(self, train_set, normalize=True):
        images, labels = train_set

        # find mean codes for each label
        codes = dbn.encode(images)
        categories = np.unique(labels)
        vocab = []
        for category in categories:
            pointer = codes[labels == category].mean(0)
            vocab.append(pointer)

        vocab = np.array(vocab, dtype=codes.dtype)
        if normalize:
            vocab /= norm(vocab, axis=1, keepdims=True)

        return categories, vocab

    def backprop(self, train, test, n_epochs=30):
        categories = np.unique(train[1])  # unique labels

        # --- compute backprop function
        dtype = self.rbms[0].dtype
        rate = tt.cast(10., dtype)

        batch = tt.matrix('batch', dtype=dtype)
        targets = tt.matrix('targets', dtype=dtype)

        if self.classifier is None:
            classifier0 = np.random.normal(
                size=(self.rbms[-1].n_hid, 10)).astype(dtype)
        else:
            classifier0 = self.classifier
        classifier = theano.shared(classifier0, name='classifier')

        # compute coding error
        codes = self.propup(batch)
        outputs = tt.nnet.softmax(tt.dot(codes, classifier))
        rmses = tt.sqrt(tt.sum((outputs - targets)**2, axis=1))
        # rmses = tt.sum(tt.abs_(outputs - targets), axis=1)
        error = tt.mean(rmses)

        # compute gradients
        params = [classifier]
        for rbm in self.rbms:
            params.extend([rbm.W, rbm.c])

        updates = collections.OrderedDict()
        grads = tt.grad(error, params)
        for param, grad in zip(params, grads):
            updates[param] = param - rate * grad

        train_dbn = theano.function([batch, targets], error, updates=updates)

        # --- find target codes
        images, labels = train

        targets = np.zeros((images.shape[0], 10), dtype=dtype)
        for i, category in enumerate(categories):
            target = np.zeros(10)
            target[i] = 1
            targets[category == labels] = target

        # --- begin backprop
        batch_size = 1000

        ibatches = images.reshape(-1, batch_size, images.shape[1])
        tbatches = targets.reshape(-1, batch_size, targets.shape[1])

        for epoch in range(n_epochs):
            costs = []
            for batch, target in zip(ibatches, tbatches):
                costs.append(train_dbn(batch, target))

            self.classifier = classifier.get_value()
            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

    def test(self, train_set, test_set):
        # find vocabulary pointers on training set
        categories, vocab = self.get_categories_vocab(train_set)

        # encode test set and compare to vocab pointers
        images, labels = test_set
        codes = self.encode(images)
        dots = np.dot(codes, vocab.T)
        tlabels = categories[np.argmax(dots, axis=1)]
        errors = (tlabels != labels)
        return errors


# --- load the data
filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)

# --- pretrain with CD
shapes = [(28, 28), 500, 200, 50]
n_layers = len(shapes) - 1
rf_shapes = [(9, 9), None, None]
hidlinear = [False, False, True]
rates = [0.1, 0.1, 0.001]
assert len(rf_shapes) == n_layers
assert len(hidlinear) == n_layers
assert len(rates) == n_layers

images, labels = train
test_images = images[:100]
n_epochs = 15
batch_size = 100

dbn = DBN()
data = images
for i in range(n_layers):
    savename = "layer%d.npz" % i
    if not os.path.exists(savename):
        batches = data.reshape(
            data.shape[0] / batch_size, batch_size, data.shape[1])

        rbm = RBM(shapes[i], shapes[i+1],
                  rf_shape=rf_shapes[i], hidlinear=hidlinear[i])
        dbn.rbms.append(rbm)
        rbm.pretrain(batches, dbn, test_images,
                     n_epochs=n_epochs, rate=rates[i])
        rbm.save(savename)
    else:
        rbm = RBM.load(savename)
        dbn.rbms.append(rbm)

    data = rbm.encode(data)


plt.figure(99)
plt.clf()
recons = dbn.reconstruct(test_images)
plotting.compare([test_images.reshape(-1, 28, 28),
                  recons.reshape(-1, 28, 28)],
                 rows=5, cols=20)

print "mean error", dbn.test(train, test).mean()

# --- train with backprop
dbn.backprop(train, test, n_epochs=100)
