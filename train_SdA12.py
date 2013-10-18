# -*- coding: utf-8 -*-
"""
Trains SdA that outputs chroma vectors

Created on Sat Feb 16 01:27:16 2013

@author: Nikolay
"""

import theano.tensor as T
from base_train_SdA import SdATrainer
from Utils.util import to_chroma, asarray

class SdATrainerChroma(SdATrainer):
    def prepare_chords(self, chords):
        return map(lambda x: to_chroma(x), chords)

    def chords_to_array(self, chords):
        return asarray(chords)
    
def main(ins, layers_sizes, recurrent_layer,
         model_file = 'model/sda.dat',
         train_file = 'sda.csv',
         layers_file = 'layers.dat'):
    tr = SdATrainerChroma(ins = ins,
                          layers_sizes = layers_sizes,
                          outs = 12,
                          log_activation = T.tanh,
                          sda_file = model_file,
                          train_file = train_file,
                          layers_file = layers_file,
                          recurrent_layer = recurrent_layer)
    tr.train_SdA()

if __name__ == '__main__':
    main()
