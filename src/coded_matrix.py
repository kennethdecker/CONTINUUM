import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

'''
This function creates a full factorial coded matrix for a set of alternatives. The input is a 1 x n array which tells the number
of levels of each alternative. The length of the input array is equivalent to the number of factors, and each value in the 
array corresponds to number of levels avaiable for that factor.

'''

def coded_matrix(levels):

    m = np.prod(levels)
    n = np.sum(levels)
    design = np.zeros((m,n))

    categories = len(levels)

    ssize = m
    ncycles = ssize
    start = 0

    for i in range(categories):

        # settings = np.array(range(levels[i])) + 1
        coded = np.diag(np.matrix([1]*levels[i]).A1)
        nreps = ssize/ncycles
        ncycles = ncycles/levels[i]
        # settings = np.repeat(settings, nreps, axis = 0)
        coded = np.repeat(coded, nreps, axis = 1)
        # settings = np.reshape(settings, (-1,1))
        coded = coded.T
        # settings = np.repeat(settings, ncycles, axis = 1)
        # settings = np.reshape(settings.T, (-1,1))
        new_coded = coded
        for j in range(ncycles-1):
            new_coded = np.vstack((new_coded, coded))

        design = design.T
        design[start:start + levels[i],:] = new_coded.T
        start = start + levels[i]
        design = design.T

    return design
    

if __name__ == '__main__':

    levels = np.array([2,3,2])
    print coded_matrix(levels)
