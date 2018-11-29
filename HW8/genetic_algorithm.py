import numpy as np
import random
import operator
import copy

def rand_bin_str(length):
    ''' This is a helper function for the
        generate function.
    '''
    binstr = ""
    for l in range(length):
        binstr += random.choice(["0","1"])
    return binstr

def generate(parts, solutions):
    ''' This function will generate a number 
        of randon solutions as the first 
        generation of the GA'''
    rand_solutions = []
    for s in range(solutions):
        indv_solution = []
        for p in range(parts):
            indv_solution.append(rand_bin_str(parts))
        rand_solutions.append(indv_solution)
    return rand_solutions

def mutate(gene, p):
    ''' This function iterates over each
        value and inverts it, if the random
        number is less than the porbability
        (p) of a mutation.
    '''
    for strg in range(len(gene)):
        c = 0
        gAcc = list(gene[strg])
        for g in range(len(gAcc)):
            if random.uniform(0, 1) < p:
                c += 1
                if gAcc[g] == '0':
                    gAcc[g] = '1'
                else:
                    gAcc[g] = '0'
        gene[strg] = ''.join(gAcc)
    return gene, c
            

def cross_over(gene1, gene2):
    ''' This function takes two genes and swaps
        all values after a certain index.
    '''
    gene1 = ''.join(gene1)
    gene2 = ''.join(gene2)
    gene1 = list(gene1)
    gene2 = list(gene2)
    index = random.randint(0, len(gene1)-1)
    swp1 = gene1[index:]
    swp2 = gene2[index:]
    gene1[index:] = swp2
    gene2[index:] = swp1
    gene1 = ''.join(gene1)
    gene2 = ''.join(gene2)
    gene1 = [gene1[i:i+4] for i in range(0, len(gene1), 4)]
    gene2 = [gene2[i:i+4] for i in range(0, len(gene2), 4)]
    return gene1, gene2


def fit(y, xs):
    ''' This function will return
        an array which contains the index 
        of a soluton and the fit associated
        with that solution.'''
    scores = []
    i = 0
    print("Solution" + ' '*25 + "Decimal" + ' ' * 10 + "Y-Value" + ' '*10 + "Fitness Value")
    for x in xs:
        value = 0
        for xi in x:
            value += int(xi, 2)
        score = abs(y-value)
        print(str(x) + ' ' * 8 + str(value) + ' '*12 + str(y) + ' '*10 + str(score))
        scores.append((i, score))
        i+=1
    return scores

target = random.randint(0, 60)
solutions = 20
parts = 4
p = 0.1
iterations = 1000

# Generate the first generation
gen1 = generate(parts, solutions)
cgen = copy.deepcopy(gen1)
mutation_count = 0
crossOver_count = 0

# begin iterating through generations
for i in range(iterations):
    print("Generation {0}".format(i))
    scores = fit(target, cgen)
    scores.sort(key=operator.itemgetter(1))
    top_ten_index = scores[:10]
    low_ten_index = scores[10:]
    top_ten = [cgen[index[0]] for index in top_ten_index]
    low_ten = [cgen[index[0]] for index in low_ten_index]

    #print("Top Ten Fit")
    #sctt = fit(target, top_ten)
    #print("Low Ten Fit")
    #sclt = fit(target, low_ten)
    
    cgen = top_ten
    children = copy.deepcopy(top_ten)
    for j in range(len(children)):
        children[j], c = mutate(children[j], p)
        mutation_count += c
    for j in range(0, 10):
        gi1 = 0
        gi2 = 0
        gi1 = random.randint(0, len(children)-1)
        gi2 = random.randint(0, len(children)-1)
        g1, g2 = cross_over(children[gi1], children[gi2])
        children[gi1] = g1
        children[gi2] = g2
        crossOver_count += 1
    for j in range(len(children)):
        children[j], c = mutate(children[j], p)
        mutation_count += c
    cgen += children

print()
print()
print("Target Vaue:")
print(target)
print("Generations:")
print(iterations)
print("Mutations:")
print(mutation_count)
print("Cross Overs:")
print(crossOver_count)
print("Results:")
fit(target, cgen)
print("...")
print()