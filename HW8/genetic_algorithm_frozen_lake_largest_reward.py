import numpy as np
import gym
import random
import copy
import operator
import warnings

env = gym.make('FrozenLake-v0')
warnings.filterwarnings("ignore")

def generate(solutions, parts, actions):
    gen1 = []
    for s in range(solutions):
        gene = []
        for p in range(parts):
            gene.append(random.randint(0,actions-1))
        gen1.append(gene)
    return gen1

def printGeneration(gen):
    for gene in gen:
        print(gene)

def mutateGene(gene, actions, p):
    c = 0
    for i in range(len(gene)):
        if random.uniform(0, 1) < p:
            gene[i] = random.choice(range(actions))
            c += 1
    return gene, c

def mutateChildren(children, actions, p):
    c = 0
    for i in range(len(children)):
        children[i], ci = mutateGene(children[i], actions, p)
        c += ci
    return children, c

def crossOverGenes(gene1, gene2):
    index = random.randint(0, len(gene1)-1)
    swp1 = gene1[index:]
    swp2 = gene2[index:]
    gene1[index:] = swp2
    gene2[index:] = swp1
    return gene1, gene2

def crossOverChildren(children, count):
    c = 0
    for _ in range(count):
        gi1 = random.randint(0, len(children)-1)
        gi2 = random.randint(0, len(children)-1)
        cg1, cg2 = crossOverGenes(children[gi1], children[gi2])
        children[gi1] = cg1
        children[gi2] = cg2
        c += 1
    return children, c

def fitPolicy(policy):
    num_episodes = 500
    jList = []
    rList = []
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            j+=1
            a = policy[s]
            s1,r,d,_ = env.step(a)
            rAll += r
            s = s1
            if d == True:
                if r > 0:
                    jList.append(j)
                break
        rList.append(rAll)
    return np.mean(rList), round(np.mean(jList))

def fitChildren(children, output = False, greedy = True):
    scores = []
    rewards = []
    steps = []
    if output:
        print('Policy:' + ' '*50 + 'Reward:' + ' '*8 + 'Steps:')
    for g in range(len(children)):
        s1, s2 = fitPolicy(children[g])
        rewards.append(s1)
        steps.append(s2)
        if output:
            print(str(children[g]) + ' '*10 + str(s1) + ' '*10 + str(s2))
        scores.append((g, s1, s2))
    if greedy:
        scores.sort(key=operator.itemgetter(1))
        top_ten_index = scores[10:]
    else:
        scores.sort(key=operator.itemgetter(2))
        top_ten_index = scores[:10]
    if output:
        print('-'*80)
        print('Average:' + ' '*50 + str(round(np.mean(rewards), 3)) + ' '*10 + str(round(np.mean(steps), 3)))
        print('-'*80)
    top_ten = [children[index[0]] for index in top_ten_index]
    return top_ten

solutions = 20
parts = 16
actions = 4
mutation_factor = 0.1
iterations = 1000

gen1 = generate(solutions, parts, actions)
cgen = copy.deepcopy(gen1)

mutationCount = 0
crossOverCount = 0

for i in range(iterations):
    outout = False
    if i %10 == 0:
        outout = True
        print("Generation: {0}".format(i))
    top_ten = fitChildren(cgen, outout)
    cgen = copy.deepcopy(top_ten)
    children = copy.deepcopy(top_ten)
    children, c = mutateChildren(children, actions, mutation_factor)
    mutationCount += c
    children, c = crossOverChildren(children, 10)
    crossOverCount += c
    cgen += copy.deepcopy(children)
print()
print()
print("Generations: {0}".format(iterations))
print("Mutations: {0}".format(mutationCount))
print("Cross overs: {0}".format(crossOverCount))
print("Results:")
fitChildren(cgen, True)
print("...")
print()