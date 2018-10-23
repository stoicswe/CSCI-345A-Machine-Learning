# -*- coding: iso-8859-15 -*-

import re, math, random # regexes, math functions, random numbers
import matplotlib.pyplot as plt # pyplot
import numpy as np
from collections import defaultdict, Counter
from functools import partial, reduce
from quantum_dot_product_test import dot, distance

def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

#def dot(v, w):
#    """v_1 * w_1 + ... + v_n * w_n"""
#    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    v_norm = np.linalg.norm(v)
    v = v / v_norm
    return dot(v, v)
    # this is also a quantum call here. can compute the dot
    # of a list of vectors (v)

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

#def distance(v, w):
#   return math.sqrt(squared_distance(v, w))
   # this is where the quantum function should be called
