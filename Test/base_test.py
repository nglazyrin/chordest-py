# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:24:51 2013

@author: Nikolay
"""

import os

class TestIterator(object):
    
    def __init__(self, input_dir, output_dir, model_file):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.model_file = model_file
        self.model = self.load_model()

    def load_model(self, source, target):
        raise NotImplementedError("error message")

    def process_file(self, source, target):
        raise NotImplementedError("error message")
    
    def getFirstColumns(self, data, offset, cols):
        return [x[offset:offset+cols] for x in data]
    
    def iterate(self):
        files_processed = 0
        for dirname, dirnames, filenames in os.walk(self.input_dir):
            for filename in filenames:
                if filename.endswith('.csv'):
                    source = os.path.join(dirname, filename)
                    target = os.path.join(self.output_dir, filename)
                    
                    self.process_file(source, target)
                    
                    files_processed = files_processed + 1
                    if files_processed % 10 == 0:
                        print (str(files_processed) + ' files processed')
        print('Done')
