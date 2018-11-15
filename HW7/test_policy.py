import gym
import numpy as np
#Assume you create the value_iteration function in a file named Value_Iteration.py
from markov_optimize_rule_frozenlake_valueIteration import  value_iteration
from markov_optimize_rule_frozenlake import policy_iteration

def policy_test(num_episodes, policy):
    #Assume the value_iteration function only takes the env, but you may have other paramters such as gamma
    #Assume the value_iteration function returns the policy in the first place, but you may return others
    #policy = value_iteration(env.env)
    wins = 0
    total_reward = 0
    for episode in range(num_episodes):
        state = env.env.reset()
        done, num_a = False, 0
        while num_a < 100:
            action = np.argmax(policy[state])
            next_state, reward, done, _ = env.env.step(action)
            #env.render()
            num_a += 1  # increment actions taken
            #total_reward += reward  # increment reward received
            state = next_state  # set current state to next state
            # terminate if we're done and increment `wins`
            if done:
                wins += 1
                break

                    
    print("Num wins", wins, "out of %d\n" % num_episodes)

env = gym.make('FrozenLake-v0')
env.reset()
num_episodes = 1000

print("----------------")
print("Value Iteration:")
print("----------------")
policy = value_iteration(env.env)
policy_test(num_episodes, policy)
print("============================")
print()
print("----------------")
print("Policy Iteration")
print("----------------")
policy = policy_iteration()
policy_test(num_episodes, policy)