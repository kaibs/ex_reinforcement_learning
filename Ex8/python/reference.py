import gym
import numpy as np
from copy import copy



def getAMax(env, s, gamma, V_states, n_actions):

    max_v = 0
    max_a = 0

    # loop over actions
    for i in range(n_actions):
        
        maxVal = 0

        # loop over states
        for j in range(len(env.P[s][i])):
            maxVal += env.P[s][i][j][0]*(env.P[s][i][j][2] + gamma*(1-env.P[s][i][j][3])*V_states[int(env.P[s][i][j][1])])

        # check if new max
        if maxVal > max_v:
            max_v = maxVal
            max_a = i

    return max_v, max_a


def value_iteration(env):

    # Init some useful variables:
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8

    step_cntr = 0
    # loop until converge
    while True:

        step_cntr += 1

        # copy old state-vals
        v = copy(V_states)

        # loop over states
        for i in range(n_states):

            # get greedy action
            V_states[i], _ = getAMax(env, i, gamma, V_states, n_actions)

        # check break-cond
        if np.max(np.abs(v - V_states)) <= theta:
                break

    # get resulting policy
    _, pol = zip(*[getAMax(env, s, gamma, V_states, n_actions) for s in range(n_states)])

    return pol, V_states