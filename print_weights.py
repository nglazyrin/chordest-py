# -*- coding: utf-8 -*-
"""
Created on Sat Apr 06 09:53:49 2013

@author: Nikolay
"""

import cPickle
import theano
import matplotlib.pyplot as p

prefix = 'E:/personal/dissertation/sda.48-36/model/'
with open(prefix + 'sda0.dat', 'rb') as f:
    (da, logl, sigmoid) = cPickle.load(f)
with open(prefix + 'layers0.dat', 'rb') as f:
    (da_layers, sigmoid_layers) = cPickle.load(f)
idx0 = 0
idx1 = 16
s = da[idx0].W[idx1]
s0 = da_layers[idx0].W[idx1]
#s0 = sigmoid.W[15]
f = theano.function([], s)
f0 = theano.function([], s0)
y = f()
y0 = f0()
x = range(len(y))
x0 = range(len(y0))
p.subplots = 2
p.plot(x,y)
p.plot(x0,y0)
p.legend(["autoencoder", "final"])
p.show()