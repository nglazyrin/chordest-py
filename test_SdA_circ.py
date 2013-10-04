# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 00:30:12 2013

@author: Nikolay
"""

import csv
import cPickle
import theano
import numpy

from DL.SdA_modified import SdA
from Test.base_test import TestIterator
from Utils import util
import base_train_SdA

class test_SdA_circ(TestIterator):
    def load_model(self):
        with open(self.model_file, 'rb') as f:
            (dA_layers, sigmoid_layers, log_layer) = cPickle.load(f)
        return self.restore_sda(dA_layers, sigmoid_layers, log_layer)

    def restore_sda(self, dA_layers, sigmoid_layers, log_layer, is_vector_y=False):
        layers = (dA_layers, sigmoid_layers)
        hidden_layers_sizes = map(lambda x: x.n_hidden, dA_layers)
        self.n_ins = dA_layers[0].n_visible
        n_outs = log_layer.W.shape[1]
        sda = SdA(n_ins=self.n_ins, n_outs=n_outs,
                  hidden_layers_sizes=hidden_layers_sizes,
                  layers=layers, log_layer=log_layer, is_vector_y=is_vector_y,
                  log_activation=theano.tensor.tanh)
        return sda

    def process_file(self, source, target):
        subnotes = 1
        with open(source, 'rb') as i:
            reader = csv.reader(i)
            (before, chords) = util.list_spectrum_data(reader,
                     components=self.n_ins + 12 * subnotes, allow_no_chord=True)
        result = None
        for offset in range(12):
            data = numpy.asmatrix(before, dtype=theano.config.floatX)
#            if extra_octave:
            start, end = offset * subnotes, offset * subnotes + self.n_ins
            data = data[:,start:end]
#            else:
#                data = numpy.roll(data, -offset * subnotes, axis=1)
            temp = self.through_sda(self.model, data)
            temp = numpy.roll(temp, offset * subnotes, axis=1)
            if result == None:
                result = numpy.zeros(temp.shape)
            result = numpy.add(result, temp)
#        result = self.through_sda(self.model, before)

#        s = result.shape[0]
#        s = 0
#        ss = numpy.zeros(shape=[s, s])
#        for i, v1 in enumerate(result):
#            for j, v2 in enumerate(result):
#                ss[i][j] = numpy.linalg.norm(self.normalize(v1) - self.normalize(v2))
#        for i in range(s):
#            eps = numpy.sort(ss[i])[int(s * 0.03)]
#            for j in range(s):
#                if ss[i][j] > eps:
#                    ss[i][j] = 1
#        result1 = numpy.zeros(shape = result.shape)
#        for i in range(s):
#            sumW = 0
#            for j in range(s):
#                w = 1 - ss[i][j]
#                result1[i] = numpy.add(result1[i], numpy.multiply(result[j], w))
#                sumW = sumW + w
#            result1[i] = numpy.multiply(result1[i], 1.0 / sumW)
            
        
        with open(target, 'wb') as o:
            writer = csv.writer(o)
            writer.writerows(result)
    
    def through_sda(self, sda, data):
        m = theano.shared(data, borrow=True)
        return sda.get_result(m)
    
    def normalize(self, v):
        v -= v.min()
        v *= (1.0/v.max())
        return v
    
#    def rotateLeft(self, data, n):
#        return [x.tolist()[n:] + x.tolist()[:n] for x in data]
#        return [x[n:] + x[:n] for x in data]

def main(input_dir = base_train_SdA.file_root + 'test',
         output_dir = base_train_SdA.file_root + 'encoded',
         model_file = 'model/sda.dat'):
    it = test_SdA_circ(input_dir, output_dir, model_file)
    it.iterate()

if __name__ == '__main__':
    main()
