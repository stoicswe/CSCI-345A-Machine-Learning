import gym
from gym import wrappers
import numpy as np

env = gym.make("FrozenLake-v0")
#env = wrappers.Monitor(env, "./results", force=True)

Q_1 = np.zeros([env.observation_space.n, env.action_space.n])
Q_2 = np.zeros([env.observation_space.n, env.action_space.n])
Q_3 = np.zeros([env.observation_space.n, env.action_space.n])
QS = [Q_1, Q_2, Q_3]
num_episodes = 20000
rList = []
gamma = 0.99
alpha = 0.85

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        action = np.argmax(Q_1[state, :] + Q_2[state, :] + Q_3[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        new_state, reward, done, _ = env.step(action)
        x = np.random.rand()
        if x > 0.6:
            # Q_1[state, action] = Q_1[state, action] + alpha * (reward + gamma * Q_2[new_state, np.argmax(Q_1[new_state, :])] - Q_1[state, action])
            Q_1[state, action] = Q_1[state, action] + alpha * (reward + gamma * ((QS[1] + QS[2]) / (len(QS) - 1))[new_state, np.argmax(Q_1[new_state, :])] - Q_1[state, action])
        elif x > 0.3:
            Q_2[state, action] = Q_2[state, action] + alpha * (reward + gamma * ((QS[0] + QS[2]) / (len(QS) - 1))[new_state, np.argmax(Q_2[new_state, :])] - Q_2[state, action])
            # Q_2[state, action] = Q_2[state, action] + alpha * (reward + gamma * Q_1[new_state, np.argmax(Q_2[new_state, :])] - Q_2[state, action])
        else:
            Q_3[state, action] = Q_3[state, action] + alpha * (reward + gamma * ((QS[0] + QS[1]) / (len(QS) - 1))[new_state, np.argmax(Q_3[new_state, :])] - Q_3[state, action])
        rAll += reward
        state = new_state
    rList.append(rAll)
    if i % 500 == 0 and i is not 0:
        print("Success rate: " + str(sum(rList) / i))

print("Success rate: " + str(sum(rList)/num_episodes))