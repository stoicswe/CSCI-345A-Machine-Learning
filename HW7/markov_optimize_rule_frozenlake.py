import gym
import gym.envs
import numpy as np
from gridworld import opt_policy_v, gridworld

def policy_evaluation(policy, env, gamma):
    envUp = env.env
    v = np.zeros([envUp.nS])
    v_old = np.ones([envUp.nS])
    delta = 1e-5
    delta_t = 1
    k = 0
    
    while delta_t > delta:
        env.reset()
        for s in range(envUp.nS):
            Vs = 0
            for action, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in envUp.P[s][action]:
                    Vs += action_prob * prob * (reward + gamma * v[next_state])
            v[s] = Vs
        delta_t = np.sum(np.abs(v - v_old))
        v_old = v.copy()
        k += 1
    print("Policy Evaluation")
    print("Number of iterations to convergence: %d\n" %k)
    return v

def policy_improvement(gamma, policy, V, env):
    envUp = env.env
    new_policy = np.zeros([envUp.nS, envUp.nA]) / envUp.nA
    for s in range(envUp.nS):
        q = np.zeros(envUp.nA)
        for a in range(envUp.nA):
            for prob, next_state, reward, done in envUp.P[s][a]:
                q[a] += prob * (reward + gamma * V[next_state])
        
        best_a = np.argwhere(q==np.max(q)).flatten()
        new_policy[s] = np.sum([np.eye(envUp.nA)[i] for i in best_a], axis=0)/len(best_a)
    
    num_policy_changes = np.sum(policy != new_policy)
    print("Policy Improvement")
    print("Number of policy changes: %d\n" %num_policy_changes)
    return new_policy, num_policy_changes

def policy_iteration():
    grid = gridworld()
    env = gym.make("FrozenLake-v0")
    envUp = env.env
    v = np.zeros([envUp.nS])

    policy = np.ones((envUp.nS, envUp.nA)) * 0.25
    gamma = 0.9

    stable_policy = False
    while stable_policy == False:
        v = policy_evaluation(policy, env, gamma)
        new_policy, num_changes = policy_improvement(gamma, policy, v, env)

        if (policy == new_policy).all():
            stable_policy = True
        else:
            policy = new_policy.copy()
    return policy

if __name__ == "__main__":
    policy_iteration()