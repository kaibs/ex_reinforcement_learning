import gym
import numpy as np
from copy import copy

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)
# TODO: Uncomment the following line to try the default map (4x4):
#env = gym.make("FrozenLake-v0")

# Uncomment the following lines for even larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states) # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.

gamma = 0.8


""" This is a helper function that returns the transition probability matrix P for a policy """
def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """
def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    # TODO: calculate and return v
    # (P, r and gamma already given)

    v = np.dot(np.linalg.inv(np.eye(P.shape[0]) - gamma*P), r)

    return v


# iterate trough every value
def recursive_policy(pol, s_idx, a, pols):
    # check if end is reached
    if s_idx >= n_states:

        pols.append(copy(pol))
        return

    # prevent error with initial s_idx=-1
    if s_idx >= 0:

        # create new pol
        pol[s_idx] = a

    # recursivly create other pols
    for a in range(n_actions):

        # get next s_idx
        if (s_idx + 1) in terminals():
            s_new = (s_idx + 2)
        else:
            s_new = (s_idx + 1)

        # call fcn recursivly
        recursive_policy(pol, s_new, a, pols)


def recursive_v2(pol_max):
    result = []

    if len(pol_max)== 0:
        result = []
    elif len(pol_max)==1:
        result = [[i] for i in range(pol_max[0] + 1)]
    else:
        # initial: [3,3,3,3,3,3,3,3,3]
        first_val = pol_max[0]
        rem_val = pol_max[1:]

        for i in range(first_val+1):

            for lst in recursive_v2(rem_val):

                result.append([i] + lst)

    return result


def bruteforce_policies():

    global opt_pol
    global opt_v
    global cntr

    terms = terminals()
    optimalpolicies = []

    pol = np.zeros(n_states)  # in the discrete case a policy is just an array with action = policy[state]
    optimalvalue = np.zeros(n_states)
    
    # TODO: implement code that tries all possible policies, calculate the values using def value_policy. Find the optimal values and the optimal policies to answer the exercise questions.
    # calc all possible pols
    #s_idx = -1
    #a_idx = 0
    #pols = []
    #recursive_policy(pol, s_idx, a_idx, pols)

    pol_max = [3]*9
    pols = recursive_v2(pol_max)

    # calc values for pols
    vals = []
    for i in range(len(pols)):
        vals.append(value_policy(pols[i]))

    # find max value pols
    max_val = 0
    max_val_idxs = []
    for i in range(len(vals)):
        comp_val = np.sum(vals[i])
        if comp_val > max_val:
            max_val = comp_val
            max_val_idxs = []
            max_val_idxs.append(i)
        elif comp_val >= max_val:
            max_val_idxs.append(i)

    # find related opt pols
    for i in max_val_idxs:
        optimalpolicies.append(pols[i])

    # delete duplicates
    optimalpolicies = list(set(map(tuple, optimalpolicies)))


    optimalvalue = vals[max_val_idxs[0]]
    print("nbr policies:")
    print(len(pols))
    print("Optimal value function:")
    print(optimalvalue)
    print("number optimal policies:")
    print(len(optimalpolicies))
    print("optimal policies:")
    print(np.array(optimalpolicies))
    return optimalpolicies



def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=int)  # 0 for all states
    policy_right = np.ones(n_states, dtype=int) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print (value_policy(policy_left))
    print("Value function for policy_right (always going right):")
    print (value_policy(policy_right))

    

    optimalpolicies = bruteforce_policies()


    # This code can be used to "rollout" a policy in the environment:
    """
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()