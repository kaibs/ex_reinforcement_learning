import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def eps_greedy_policy_v2(Q, s, epsilon=0.1):
    action_values = Q[s]
    nA = len(action_values)
    greedy_action = np.argmax(action_values)
    action_prob = np.ones(nA) * epsilon / nA
    action_prob[greedy_action] += (1 - epsilon)
    return  action_prob


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.2, num_ep=int(1e4)):
    nA = env.action_space.n
    nS = env.observation_space.n
    Q = np.zeros((nS,  nA))
    
    for i in range(num_ep):
        s = env.reset()
        done = False
        pi = eps_greedy_policy_v2(Q, s, epsilon=epsilon)
        a = np.random.choice(nA, p=pi)
        S = [s]
        A = [a]
        R = [0.0]
    
        T = float("inf")
        t = 0
        tau = 0
        while tau < T - 1:
            if t < T:
                s_, r, done, _ = env.step(a)

                S.append(s_)
                R.append(r)

                s = s_

                pi = eps_greedy_policy_v2(Q, s, epsilon=epsilon)
                a = np.random.choice(nA, p=pi)           
                
                if done:
                    T = t + 1
                else:
                    A.append(a)
            tau = t - n + 1
            if tau >= 0:
                Gm = np.array([gamma ** (i - tau - 1) for i in range((tau + 1), min(tau + n, T) + 1)])
                G = np.sum(Gm * R[(tau + 1):min(tau + n, T) + 1]) 
                if tau + n < T:
                    G += gamma**n * Q[S[tau + n]][A[tau + n]] 
                Q[S[tau]][A[tau]] += alpha * (G - Q[S[tau]][A[tau]])
            t += 1
    return Q

# online nstep Sarsa version (not use)
def nstep_sarsa_online(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    nA = env.action_space.n
    nS = env.observation_space.n
    Q = np.zeros((nS,  nA))
    C = np.zeros(nS) # count visit of every state (for computing GLIE eps)
    
    for i in range(num_ep):
        s = env.reset()
        done = False
        Gn = dict() # for tracking n-step coeff
        C = dict() # count to n to stop eligibility traces

        eps_policy = eps_greedy_policy(Q, C, epsilon_min=epsilon)
        a = np.random.choice(nA, p=eps_policy(s))

        while not done:
            # update discount factor each step 
            for sa in Gn:
                for i in range(len(Gn[sa])):    
                    Gn[sa][i] *= gamma

            # everytime we visit a state, start tracking cummulative reward
            if (s, a) not in Gn:
                Gn[(s, a)] = []
            Gn[(s, a)].append(1.0)
            C[s] += 1

            # append count C at s, a and increase other running count
            for sa in C:
                for i in range(len(C[sa])):
                    C[sa][i] += 1
            if (s, a) not in C:
                C[(s, a)] = []
            C[(s, a)].append(1)

            s_, r, done, _ = env.step(a)
            eps_policy = eps_greedy_policy(Q, C, epsilon_min=epsilon)
            a_ = np.random.choice(nA, p=eps_policy(s_))

            # update returns
            for sa in Gn:
                for i in range(len(Gn[sa])): 
                    Q[sa[0]][sa[1]] += Gn[sa][i] * r

            # remove running n step
            for sa in Gn:
                if len(C[sa]) != 0 and C[sa][0] == n:
                    C[sa].pop(0)
                    Gn[sa].pop(0)

            s, a = s_, a_
    return Q

# baseline to compare different setting on n-step sarsa
def mc_control_on_policy(env, alpha=0.1, gamma=0.9, num_ep=int(1e4)):
    nA = env.action_space.n # action set size of env
    nS = env.observation_space.n
    C = np.zeros(nS) # count visit of every state (for computing GLIE eps)
    Q = np.zeros((nS,  nA)) # action-value func
    eps_policy = eps_greedy_policy(Q, C)  # eps policy
    
    for episode in range(1, num_ep + 1):
        state = env.reset()
        
        done = False
        eligibility_traces = defaultdict(float) # for tracking return of visited states
        returns = defaultdict(float) # return of each state-action pair
        
        while not done:
            action = np.random.choice(nA, p=eps_policy(state))
            state_action = (state, action)
            
            # update routine of discount factor 
            for _state_action in eligibility_traces:
                eligibility_traces[_state_action] *= gamma
            
            # everytime we visit a state, increase one count
            eligibility_traces[state_action] += 1.0
            C[state] += 1
            
            next_state, reward, done, _ = env.step(action)
            
            for _state_action in eligibility_traces:
                returns[_state_action] += eligibility_traces[_state_action] * reward
            
            state = next_state
        
        # update action-value func following GLIE schedule
        for state, action in returns:
            Q[state][action] += alpha * (returns[(state, action)] - Q[state][action])
        
        # improve policy
        eps_policy = eps_greedy_policy(Q, C)
    return Q

def plot(data, N, alphas, num_ep, annotation=""):
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')

    for i, line in enumerate(data):
        plt.plot(alphas, line, color=palette(i), linewidth=1, alpha=0.9, label="n={}".format(N[i]))
    plt.legend(loc="best", ncol=2)

    plt.xlabel("alpha")
    plt.ylabel("RMS Error on first {} episodes".format(num_ep))

    plt.show()


if __name__ == "__main__":
    env = gym.make('FrozenLake-v0', map_name="4x4")

    num_ep = int(1e2)
    num_N = 6
    N = [2**i for i in range(num_N)]
    alphas = np.linspace(0, 1, 30)
    data = [[] for _ in range(num_N)]

    # calculate baseline
    Q_base = mc_control_on_policy(env, num_ep=int(1e4))

    for i, n in enumerate(N):
        for alpha in alphas:
            Q = nstep_sarsa(env, n=n, alpha=alpha, num_ep=num_ep)
            data[i].append(np.linalg.norm(Q - Q_base))

    plot(data, N, alphas, num_ep=num_ep)
