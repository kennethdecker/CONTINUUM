import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

'''

This function performs topsis on an array of data. Function must pe provided a 2d array in which columns correspond to attributes and rows
correspond to alternatives. Values of each element is the value that an alternative has with respect of a given attribut. A scaling array 
must also be provided that contains the relative weighing factors of each attribute. Finally, decision array must be provided to tell whether
values for each attribute should be maximized or minimized.

Inputs
-----
data_array: 2d array of floats
    data array is the matrix of data that TOPSIS is being performed on. Columns correspond to attributes and rows correspond to aalternatives
scaling_array: row array of floats
    scaling array tells weighing factors for importance of each alternative. Each position in scaling array corresponds to same column of data_array
decision_array: row array of ints
    decision array tells whether attribute values should be maximixed or minimized. "1" corresponds to maximization, "0" corresponds to minimization

Returns
-----
ideal: int
    Returns the index of the row corresponding to the ideal alternative
data_array[ideal]: row array of floats
    Returns row of attribute values of ideal alternative

'''

def run_topsis(data_array, scaling_array, decision_array):
    
    m,n = data_array.shape

    norm_array = np.zeros((1,n), dtype = np.float64)
    pos_ideal = np.zeros((1,n), dtype = np.float64)
    neg_ideal = np.zeros((1,n), dtype = np.float64)
    pos_dist = np.zeros((1,n), dtype = np.float64)
    neg_dist = np.zeros((1,n), dtype = np.float64)
    closeness = np.zeros((1,n), dtype = np.float64)

    trans = data_array.T
    for i in range(n):
        norm_array[0][i] = np.linalg.norm(trans[i])
        
    norm_data = data_array/norm_array

    if np.sum(scaling_array) != 1.0:
        raise ValueError('Sum of scaling array must equal 1')

    norm_data = norm_data*scaling_array
    trans = norm_data.T

    for i in range(n):
        if decision_array[i] == 1:

            pos_ideal[0][i] = np.max(trans[i])
            neg_ideal[0][i] = np.min(trans[i])

        elif decision_array[i] == 0:

            pos_ideal[0][i] = np.min(trans[i])
            neg_ideal[0][i] = np.max(trans[i])

        else:
            raise ValueError('Decision Array can only contain 1 (maximize) or 0 (minimize)')

    for i in range(n):

        pos_dist[0][i] = np.sqrt(np.sum((norm_data[i] - pos_ideal)**2.0))
        neg_dist[0][i] = np.sqrt(np.sum((norm_data[i] - neg_ideal)**2.0))
        closeness[0][i] = neg_dist[0][i]/(neg_dist[0][i] + pos_dist[0][i])

    ideal = np.argmax(closeness)

    return (ideal, data_array[ideal])




if __name__ == '__main__':

    test_array = np.array([[1,2],[3,4]], dtype = np.float64)
    scaling_array = np.array([.4,.6], dtype = np.float64)
    decision_array = np.array([1,0], dtype = np.int)

    i, vals = run_topsis(test_array, scaling_array, decision_array)

