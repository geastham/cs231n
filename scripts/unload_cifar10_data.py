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

# Unload training data
def unload_training_data():
    # load first batches
    file = os.path.dirname(os.path.abspath(__file__)) + "/../data/cifar-10-batches-py/data_batch_1"

    # Unpickle the data batch
    data = unpickle(file)

    # Merge with labels
    training_data = np.concatenate((np.array(data['labels']).reshape(10000, 1), data['data']), axis=1)

    # iterate remaining 4 training batches and concatenate to starting matrix
    for i in range(4):
        # Get system file location
        file = os.path.dirname(os.path.abspath(__file__)) + "/../data/cifar-10-batches-py/data_batch_" + str(i+2)

        # Unpickle the data batch
        data = unpickle(file)

        # Merge with labels
        merged_data = np.concatenate((np.array(data['labels']).reshape(10000, 1), data['data']), axis=1)

        # Append to all training data
        training_data = np.concatenate((training_data, merged_data), axis=0)

    print(training_data)

    # Save data to txt
    np.savetxt(os.path.dirname(os.path.abspath(__file__)) + '/../data/cifar10-train.txt', training_data, fmt='%i')

# Unload test data
def unload_test_data():
    # Get system file location
    file = os.path.dirname(os.path.abspath(__file__)) + "/../data/cifar-10-batches-py/test_batch"

    # Unpickle the data batch
    data = unpickle(file)

    # Merge with labels
    merged_data = np.concatenate((np.array(data['labels']).reshape(10000, 1), data['data']), axis=1)

    print(merged_data)

    # Save data to txt
    np.savetxt(os.path.dirname(os.path.abspath(__file__)) + '/../data/cifar10-test.txt', merged_data, fmt='%i')

def main():

    # Unload training data
    #unload_training_data()

    # Unload test data
    unload_test_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except:
        logging.exception("unload_cifar10_data.py")
        sys.exit(1)
