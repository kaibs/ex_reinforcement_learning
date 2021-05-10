import gym
import numpy as np
from copy import copy

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

def getAMax(s, gamma, V_states):

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


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r

    step_cntr = 0
    # loop until converge
    while True:

        step_cntr += 1

        # copy old state-vals
        v = copy(V_states)

        # loop over states
        for i in range(n_states):

            # get greedy action
            V_states[i], _ = getAMax(i, gamma, V_states)

        # check break-cond
        if np.max(np.abs(v - V_states)) <= theta:
                break

    # get resulting policy
    _, pol = zip(*[getAMax(s, gamma, V_states) for s in range(n_states)])

    print("converged in " + str(step_cntr) + " steps")
    print("\n")
    print("optimal value fcn:")
    print(str(V_states))
    print("\n")

    return pol




def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()