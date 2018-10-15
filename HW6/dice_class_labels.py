import numpy as np
import random
import math

def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v,w))

def sum_of_squares(v):
    return dot(v,v)

def mean(x):
    return sum(x) / len(x)

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / n

def expected_value(p, x):
    return sum([pi * xi for pi, xi in zip(p,x)])

roll_count = 1000
for dice_count in range(2, 6):
    tc = {}
    for i in range(roll_count):
        total = 0
        for k in range(dice_count):
            total += random.randint(1,6)
        if total in tc:
            tc[total] += 1
        else:
            tc[total] = 1
    ck = tc.keys()
    for c in ck:
        tc[c] = tc[c] / roll_count
    entropy = 0
    for k in ck:
        entropy += (-tc[k] * math.log(tc[k], 2))
    #print("Probabilities for {0} dice: {1}".format(dice_count, tc))
    print("Entropy for {0} dice: {1}".format(dice_count, entropy))

print("Using second algorithm:")
#### Second algorithm ####
for dice_count in range(2, 6):
    tc = {}
    lbls = []
    for i in range(roll_count):
        total = 0
        for k in range(dice_count):
            total += random.randint(1,6)
        lbls.append(total)
        #if total in tc:
        #    tc[total] += 1
        #else:
        #    tc[total] = 1
    for l in lbls:
        if l in tc:
            tc[l] += 1
        else:
            tc[l] = 1
    ck = tc.keys()
    for c in ck:
        tc[c] = tc[c] / len(tc.keys())
    entropy = 0
    for k in ck:
        entropy -= (tc[k] * math.log(tc[k], 2))
    #print("Probabilities for {0} dice: {1}".format(dice_count, tc))
    print("Entropy for {0} dice: {1}".format(dice_count, entropy))