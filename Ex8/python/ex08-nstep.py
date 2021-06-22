import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from reference import value_iteration



# epsilon greedy policy
def greedy_pol(Q, s, epsilon):

    a_vals = Q[s]
    nbr_a = len(a_vals)

    greedy_a = np.argmax(a_vals)
    probs = np.ones(nbr_a)*epsilon/nbr_a
    probs[greedy_a] += (1-epsilon)

    return probs


def get_pol(env, Q, s, epsilon):

    # explore
    if random.random() < epsilon:

        a = env.action_space.sample()

    # exploit
    else:

        a = np.argmax(Q[s])

    return a



def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.3, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """

    # init params
    nbr_a = env.action_space.n
    nbr_s = env.observation_space.n
    #Q = np.zeros((nbr_s,  nbr_a))
    Q = np.ones((nbr_s,  nbr_a))*0.1

    # go over episodes
    for i in range(num_ep):
        
        # reset/init
        s = env.reset()
        done = False
        t = 0
        tau = 0
        T = float("inf")

        a = get_pol(env, Q, s, epsilon)

        rew = [0.0]
        S = [s]
        A = [a]

        while (tau < T-1):

            if t < T:

                s_new, r, done, _ = env.step(a)
                S.append(s_new)
                rew.append(r)

                s = s_new
                a = get_pol(env, Q, s, epsilon)
                
                if done:
                    T = t+1
                else:
                    A.append(a)

            tau = t - n + 1


            if tau >= 0:

                G_idx = np.array([gamma ** (j - tau - 1) for j in range((tau + 1), min(tau + n, T) + 1)])
                G = np.sum(G_idx*rew[(tau + 1):(min(tau + n, T) + 1)])

                if tau + n < T:
                    G += (gamma**n)*Q[S[tau + n]][A[tau + n]]

                Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])

            t += 1


    V = np.array([np.max(Q[i][:]) for i in range(np.shape(Q)[0])])


    return Q, V


        
if __name__ == "__main__":

    # init env
    env=gym.make('FrozenLake-v0', map_name="8x8")

    # params
    num_ep = 100000
    n_arr = [1, 2, 4, 8, 16, 64, 128, 512]
    #n_arr = [1, 128, 512]
    #alpha_arr = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    alpha_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = [[] for n in range(len(n_arr))]

    # get reference with value iteration
    _, V_ref = value_iteration(env)

    # loop over params
    for i in range(len(n_arr)):
        for a in alpha_arr:
            print("running n=" + str(n_arr[i]) + "  and  alpha=" + str(a))
            #Q, Q_ref = nstep_sarsa(env, n=n_arr[i], alpha=a, num_ep=num_ep)
            Q, V = nstep_sarsa(env, n=n_arr[i], alpha=a, num_ep=num_ep)
            #Q_ref = nstep_sarsa(env, n=n_arr[i], alpha=a, num_ep=10)
            results[i].append(np.linalg.norm(V - V_ref)**2)

    # plot results
    plt.rc('legend', fontsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    fig = plt.figure(1)
    for i in range(len(n_arr)):
        plt.plot(alpha_arr, results[i], label=n_arr[i], linewidth=2.0)
    plt.legend()
    plt.xlabel("alpha", fontsize=18)
    plt.ylabel("error", fontsize=18)
    plt.show()