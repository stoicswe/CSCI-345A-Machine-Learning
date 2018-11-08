import gym
import numpy as np
from gridworld import opt_policy_v, gridworld

# Policy Evaluation Step
def policy_evaluation(policy, grid, gamma):
    
    # Initialize V(s)
    v = np.zeros(grid.dim)
    # Set terminal state values
    v[grid.pos_goal[0], grid.pos_goal[1]] = grid.rew_goal
    v[grid.pos_trap[0], grid.pos_trap[1]] = grid.rew_trap
    v_reset = v.copy()
    delta = 1e-5
    delta_t = 1
    k = 0
    
    while delta_t > delta:
        v_new = v_reset.copy()
        for i in range(grid.dim[0]):
            for j in range(grid.dim[1]):
                # Check for terminal state
                if [i, j] == grid.pos_goal or [i, j] == grid.pos_trap:
                    continue
                else:
                    for prob, action in zip(policy[i, j],
                                            grid.action_space):
                        grid.s = [i, j]
                        s_1, r, _ = grid.action(action)
                        prob = grid.action_prob[action]
                        v_new[i, j] += prob * (r + gamma * 
                                               v[s_1[0], s_1[1]])
        k += 1
        delta_t = np.sum(np.abs(v - v_new))
        v = v_new.copy()
    print("Policy Evaluation")
    print("Number of iterations to convergence: %d\n" %k)
    return v

def policy_improvement(policy, values, grid):
    new_policy = np.zeros(policy.shape)
    
    for i in range(grid.dim[0]):
        for j in range(grid.dim[1]):
            # Calculate the value for each action
            action_vals = []
            for a_num, a in enumerate(grid.action_space):
                grid.s = [i, j]
                s_1, r, complete = grid.action(a)
                v_ = r + gamma * values[s_1[0], s_1[1]]
                action_vals.append([v_, a_num])
            
            # Select the best action
            action_vals = np.array(action_vals)
            # For cases where there are multiple "best actions"
            # define their selection probabalistically
            act_max = np.max(action_vals[:,0])
            best_actions = action_vals[np.where(
                    action_vals[:,0]==act_max), 1].flatten()
            for act in best_actions:
                new_policy[i, j, int(act)] = 1 / len(best_actions)
            
    num_policy_changes = np.sum(policy != new_policy)
    print("Policy Improvement")
    print("Number of policy changes: %d\n" %num_policy_changes)
    return new_policy, num_policy_changes

grid = gridworld()

policy = policy = np.ones((grid.dim[0], grid.dim[1], 
                  len(grid.action_space))) * 0.25
gamma = 0.9

stable_policy = False
while stable_policy == False:
    v = policy_evaluation(policy, grid, gamma)
    print(v.round(1))
    new_policy, num_changes = policy_improvement(policy, v, grid)
    
    # Check for stability
    if num_changes == 0:
        stable_policy = True
    else:
        policy = new_policy.copy()