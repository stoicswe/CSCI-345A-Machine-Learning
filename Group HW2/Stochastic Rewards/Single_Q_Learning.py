import gym
import numpy as np

import frozen_lake
import core
from core import Wrapper
import random
#gamma = .95
#alpha = .8

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 20000
gamma = 0.95
alpha = 0.99
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        new_state,reward,done,_ = env.step(action)
        if reward != 0:
            reward = random.uniform(0, 2)
        Q[state,action] = Q[state,action] + alpha*(reward + gamma*np.max(Q[new_state,:]) - Q[state,action])
        rAll += reward
        state = new_state
    rList.append(rAll)
    if i % 500 == 0 and i is not 0:
        print("Success rate: " + str(sum(rList) / (i)))
print("Single Q Learning")
print("Final Success rate: " +  str(sum(rList)/num_episodes))