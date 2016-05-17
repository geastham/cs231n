#!/usr/bin/env python

# -- Unload CIFAR-10 Data --
# This script unpickles the CIFAR-10 dataset (expected in /data/cifar-10-python) and
# loads the data into a format suitable for Spark / Scala ecosystem.

import os
import sys
import logging
import cPickle
import numpy as np

# Unpickles a particular batch file (return dict)
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#def load_all_data():

def main():

    # Get system file location
    file = os.path.dirname(os.path.abspath(__file__)) + "/../data/cifar-10-batches-py/data_batch_1"

    # Unpickle the data batch
    data = unpickle(file)

    # Load data
    print("Number of images: " + str(len(data['data'])))
    print("Number of labels: " + str(len(data['labels'])))

    # Examine data
    print(data['data'])

    # Examine labels
    print(np.array(data['labels']).reshape(10000, 1))

    # Merge with labels
    merged_data = zip(np.array(data['labels']).reshape(10000, 1), data['data'])

    # Save data to txt
    np.savetxt('data_1.txt', merged_data, fmt='%i')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except:
        logging.exception("unload_cifar10_data.py")
        sys.exit(1)
