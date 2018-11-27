import numpy as np
import random

target = 45 #some number as the target
solutions = 20
parts = 4

def rand_bin_str(length):
    binstr = ""
    for l in range(length):
        binstr += random.choice(["0","1"])
    return binstr

def generate(parts, solutions):
    '''This function will generate a number of randon solutions
    as the first generation of the GA'''
    rand_solutions = []
    for s in range(solutions):
        rand_solutions.append(rand_bin_str(parts))
    return rand_solutions

#def mutate(gene): #takes one and gives one

#def cross_over(gene1, gene2): #takes 2 and gives one


def fit(y, xs):
    '''This function will return
    an array which contains the fits
    for multiple solutions.'''
    scores = []
    i = 0
    for x in xs:
        value = 0
        for xi in x:
            value += int(xi, 2)
        score = y-value
        print("Solution          Score")
        print("{0}           {1}".format(x, score))
        scores.append((i, score))
        i+=1
    return scores

gen1 = generate(parts, solutions)
print(gen1)
scores = fit(target, gen1)