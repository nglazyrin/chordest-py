# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 00:43:40 2013

@author: Nikolay
"""

import os, sys
import time
import csv
import numpy
import theano
import cPickle

from DL.SdA_modified import SdA
from Utils.util import list_spectrum_data, shuffle_2_numpy

file_root = 'E:/personal/KFU/'

class MyException(Exception):
    pass

class SdATrainer(object):
    
    def __init__(self, ins, layers_sizes, outs, log_activation,
                 sda_file, train_file, layers_file,
                 is_vector_y = False, recurrent_layer = -1):
        self.train_file = train_file
        self.layers_file = layers_file
        self.sda_file = sda_file
        self.ins = ins
        self.layers_sizes = layers_sizes
        self.recurrent_layer = recurrent_layer
        self.corruption_levels = [.2, .2, .2]
        self.outs = outs
        self.pretrain_lr=0.03
        self.finetune_lr=0.01
        self.pretraining_epochs=15
        self.finetune_epochs=15
        self.training_epochs=1000
        self.batch_size=5
        self.log_activation = log_activation
        self.is_vector_y = is_vector_y

    def prepare_chords(self, chords):
        raise NotImplementedError("error message")

    def chords_to_array(self, chords):
        raise NotImplementedError("error message")

    def prepare_data(self, array):
        return array

    def read_data(self):
        print 'reading data from ' + self.train_file
        with open(self.train_file, 'rb') as f:
            reader = csv.reader(f)
            (array, chords) = list_spectrum_data(reader, components=self.ins)
        
        array = self.prepare_data(array)
        chords = self.prepare_chords(chords)
#        array = numpy.asarray(array, dtype=theano.config.floatX)
        chords = numpy.asarray(chords, dtype=theano.config.floatX)
        
        train = int(0.7 * len(array))
        test = int(0.85 * len(array))
        tr = numpy.copy(array[:train])
        tr_ch = numpy.copy(chords[:train])
        train_array = theano.shared(array[:train], borrow = True)
        train_chords = theano.shared(chords[:train], borrow = True)
        
        test_array = theano.shared(array[train:test], borrow = True)
        test_chords = theano.shared(chords[train:test], borrow = True)
        
        valid_array = theano.shared(array[test:], borrow = True)
        valid_chords = theano.shared(chords[test:], borrow = True)
        
        shuffle_2_numpy(tr, tr_ch)
        train_shuffled = theano.shared(tr, borrow = True)
        chords_shuffled = theano.shared(tr_ch, borrow = True)
       
        if self.recurrent_layer >= 0:
            return [[train_shuffled, chords_shuffled], [test_array, test_chords], \
                    [valid_array, valid_chords], [train_array, train_chords]]
        else:
            del train_array
            del train_chords
            return [[train_shuffled, chords_shuffled], [test_array, test_chords], \
                    [valid_array, valid_chords]]

    def load_layers(self):
        da = []
        sigmoid = []
        if (not os.path.isfile(self.layers_file)):
            return None
        with open(self.layers_file, 'rb') as f:
            (da, sigmoid) = cPickle.load(f)
        return (da, sigmoid)
#        return None
    
    def train_SdA(self):
        """
        Demonstrates how to train and test a stochastic denoising autoencoder.
    
        This is demonstrated on MNIST.
    
        :type learning_rate: float
        :param learning_rate: learning rate used in the finetune stage
        (factor for the stochastic gradient)
    
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining
    
        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
    
        :type n_iter: int
        :param n_iter: maximal number of iterations ot run the optimizer
    
        """
    
        layers = self.load_layers()
        datasets = self.read_data()
    
        train_shuffled, chords_shuffled = datasets[0]
#        if self.recurrent_layer >= 0:
#            train_set_x, train_set_y = datasets[3]
#        datasets = datasets[0:3]
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_shuffled.get_value(borrow=True).shape[0]
        n_train_batches /= self.batch_size
    
        # numpy random generator
        print '... building the model'
    
        # construct the stacked denoising autoencoder class
        if (layers):
            sda = SdA(n_ins=self.ins, hidden_layers_sizes=self.layers_sizes,
                      n_outs=self.outs, log_activation=self.log_activation,
                      is_vector_y=self.is_vector_y, layers=layers)
        else:
            sda = SdA(n_ins=self.ins, hidden_layers_sizes=self.layers_sizes,
                      n_outs=self.outs, log_activation=self.log_activation,
                      is_vector_y=self.is_vector_y,
                      recurrent_layer = self.recurrent_layer)
    
        #########################
        # PRETRAINING THE MODEL #
        #########################
        if (not layers):
            print '... getting the pretraining functions'
            # always use shuffled train data for pretraining
            pretraining_fns = sda.pretraining_functions(train_set_x=train_shuffled,
                                                        batch_size=self.batch_size)
        
            print '... pre-training the model'
            start_time = time.clock()
            ## Pre-train layer-wise
            for i in xrange(sda.n_layers):
                # go through pretraining epochs
                for epoch in xrange(self.pretraining_epochs):
                    # go through the training set
                    c = []
                    for batch_index in xrange(n_train_batches):
                        c.append(pretraining_fns[i](index=batch_index,
                                 corruption=self.corruption_levels[i],
                                 lr=self.pretrain_lr))
                    print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                    print numpy.mean(c)
            with open(self.layers_file, 'wb') as f:
                cPickle.dump((sda.dA_layers, sda.sigmoid_layers), f)
        
            end_time = time.clock()
        
            print >> sys.stderr, ('The pretraining code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
        ########################
        # FINETUNING THE MODEL #
        ########################
    
        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        # use non-shuffled train data for fine tuning in recurrent network
        if (self.recurrent_layer >= 0):
            datasets[0] = datasets[3]
        train_fn, validate_model, test_model = sda.build_finetune_functions(
                    datasets=datasets, batch_size=self.batch_size,
                    learning_rate=self.finetune_lr, useQuadratic=not self.is_vector_y)
    
        print '... finetuning the model'
    
        start_time = time.clock()
        
        [best_validation_loss, test_score] = self.runFineTuningLoop(
                n_train_batches, train_fn, validate_model, test_model)
    
        end_time = time.clock()
        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %
                     (best_validation_loss * 100., test_score * 100.))
        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
        with open(self.sda_file, 'wb') as f:
            cPickle.dump((sda.dA_layers, sda.sigmoid_layers, sda.logLayer), f)
        
    def runFineTuningLoop(self, n_train_batches, train_fn, validate_model, test_model):
        # early-stopping parameters
        patience = self.finetune_epochs * n_train_batches  # look as this many examples regardless
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        done_looping = False
        epoch = 0
        while (epoch < self.finetune_epochs) and (not done_looping):
            for minibatch_index in xrange(n_train_batches):
                [pre_act, minibatch_avg_cost] = train_fn(minibatch_index)
                iter = epoch * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses, axis=0)[1]
                    
                    if this_validation_loss != this_validation_loss: # check for nan
                        raise MyException("NaN in training")
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = test_model()
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
    
                if patience <= iter:
                    done_looping = True
                    break
            epoch = epoch + 1
        return [best_validation_loss, test_score]
