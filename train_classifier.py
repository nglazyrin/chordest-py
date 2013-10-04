# -*- coding: utf-8 -*-
"""
Created on Thu Aug 08 21:12:30 2013

@author: Nikolay
"""

import csv
import cPickle
import theano

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

from DL.SdA_modified import SdA
from Utils import util

train_file='E:/Dev/git/my_repository/chordest/result/train_dA_c.csv'
ins = 60

def read_data():
    with open(train_file, 'rb') as f:
        reader = csv.reader(f)
        (array, chords) = util.list_spectrum_data(reader, components=ins, allow_non_majmin=False)
#    (array, chords) = util.shuffle_2(array, chords)
#    array = prepare_data(array)
#    chords = prepare_chords(chords)
    train = int(0.7 * len(array))
    test = int(0.9 * len(array))
    train_array = array[:train]
    train_chords = chords_to_array(chords[:train])
    
    test_array = array[train:test]
    test_chords = chords_to_array(chords[train:test])
    
    valid_array = array[test:]
    valid_chords = chords_to_array(chords[test:])
    print 'finished reading data'
    
    return [[train_array, train_chords], [test_array, test_chords], \
            [valid_array, valid_chords]]

def chords_to_array(chords):
    return [util.chord_list.index(chord) for chord in chords]

def restore_sda(dA_layers, sigmoid_layers, is_vector_y=False):
    layers = (dA_layers, sigmoid_layers)
    hidden_layers_sizes = map(lambda x: x.n_hidden, dA_layers)
    n_ins = dA_layers[0].n_visible
    sda = SdA(n_ins=n_ins, n_outs=12,
              hidden_layers_sizes=hidden_layers_sizes,
              layers=layers, is_vector_y=is_vector_y,
              log_activation=theano.tensor.tanh)
    return sda

def load_model():
    with open('model/sda_layers.dat', 'rb') as f:
        (dA_layers, sigmoid_layers) = cPickle.load(f)
    return restore_sda(dA_layers, sigmoid_layers)

datasets = read_data()
sda = load_model()

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

train_set_x = util.through_sda_layers(sda, train_set_x)
valid_set_x = util.through_sda_layers(sda, valid_set_x)

print 'training'
#clf = OutputCodeClassifier(LinearSVC())
clf = RandomForestClassifier()
clf = clf.fit(train_set_x, train_set_y)

print 'done'
print 'testing...'

predictions = clf.predict(valid_set_x)
predictions = [int(pred) for pred in predictions]
common = 0
for [idx, chord] in enumerate(predictions):
    if valid_set_y[idx] == predictions[idx]:
        common = common + 1
print valid_set_x[15]
print valid_set_y[15]
print predictions[15]
print str(common * 1.0 / len(valid_set_x) * 100) + '% correct'
