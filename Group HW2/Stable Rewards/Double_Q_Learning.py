import gym
import numpy as np

env = gym.make("FrozenLake-v0")

Q_1 = np.zeros([env.observation_space.n, env.action_space.n])
Q_2 = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 20000
gamma = 0.95
alpha = 0.99
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        action = np.argmax(Q_1[state, :] + Q_2[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        new_state, reward, done, _ = env.step(action)
        x = np.random.rand()
        if  x > 0.5:
            Q_1[state, action] = Q_1[state, action] + alpha * (reward + gamma * Q_2[new_state, np.argmax(Q_1[new_state, :])] - Q_1[state, action])
        else:
            Q_2[state, action] = Q_2[state, action] + alpha * (reward + gamma * Q_1[new_state, np.argmax(Q_2[new_state, :])] - Q_2[state, action])
        rAll += reward
        state = new_state
    rList.append(rAll)
    if i % 500 == 0 and i is not 0:
        print("Success rate: " + str(sum(rList) / (i)))
print("Double Q Learning")
print("Final Success rate: " + str(sum(rList)/num_episodes))