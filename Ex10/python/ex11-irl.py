import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def generate_demonstrations(env, expertpolicy, epsilon=0.1, n_trajs=100):
    """ This is a helper function that generates trajectories using an expert policy """
    demonstrations = []
    for d in range(n_trajs):
        traj = []
        state = env.reset()
        for i in range(100):
            if np.random.uniform() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = expertpolicy[state]
            traj.append((state, action))  # one trajectory is a list with (state, action) pairs
            state, _, done, info = env.step(action)
            if done:
                traj.append((state, 0))
                break
        demonstrations.append(traj)
    return demonstrations  # return list of trajectories


def plot_rewards(rewards, env):
    """ This is a helper function to plot the reward function"""
    fig = plt.figure()
    dims = env.desc.shape
    plt.imshow(np.reshape(rewards, dims), origin='upper', 
               extent=[0,dims[0],0,dims[1]], 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(np.reshape(rewards, dims)[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def value_iteration(env, rewards):
    """ Computes a policy using value iteration given a list of rewards (one reward per state) """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V_states = np.zeros(n_states)
    theta = 1e-8
    gamma = .9
    maxiter = 1000
    policy = np.zeros(n_states, dtype=np.int)
    for iter in range(maxiter):
        delta = 0.
        for s in range(n_states):
            v = V_states[s]
            v_actions = np.zeros(n_actions) # values for possible next actions
            for a in range(n_actions):  # compute values for possible next actions
                v_actions[a] = rewards[s]
                for tuple in env.P[s][a]:  # this implements the sum over s'
                    v_actions[a] += tuple[0]*gamma*V_states[tuple[1]]  # discounted value of next state
            policy[s] = np.argmax(v_actions)
            V_states[s] = np.max(v_actions)  # use the max
            delta = max(delta, abs(v-V_states[s]))

        if delta < theta:
            break

    return policy


# calc pol by choosing most occuring a f.e. s 
def get_naive_pol(env, trajs):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    sa_pairs = []
    for s in range(n_states):
        for a in range(n_actions):
            sa_pairs.append((s,a))

    pairs = dict.fromkeys(sa_pairs, 0)

    # count pairs
    for traj in trajs:
        for pair in traj:

                pairs[pair] += 1


    max_pairs = {}

    # get highest scoring a for each s
    for s in range(n_states):

        max_a = 0
        score = 0

        for a in range(n_actions):

            if pairs[(s,a)] > score:
                max_a = a
                score = pairs[(s,a)]

        max_pairs[s] = max_a

    return max_pairs


# get probabilities for p(s'|s,a) from environment
def get_transition_probs(env):

    n_states = env.observation_space.n
    n_actions = env.action_space.n 


    # probs[s][a][s_new]
    probs = np.zeros((n_states, n_actions, n_states))

    for s in range(len(env_dist)):

        for a in range(n_actions):

            sa = env_dist[s][a]

            for prob in sa:

                probs[s][a][prob[1]] = prob[0] 

    return probs


# get distribution pi(a|s) from experience
def get_reference_dist(env, trajs):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    dist = np.zeros((n_states, n_actions))

    # create dict for {s: [a0, a1, a2, a3]}
    sa_pairs = {}
    for s in range(n_states):
            sa_pairs[s] = [0]*n_actions

    # count pairs in data
    for traj in trajs:
        for pair in traj:
                sa_pairs[pair[0]][pair[1]] += 1 

    # calc probs for dist
    for s in sa_pairs:

        total_occurances = sum(sa_pairs[s])
        
        if total_occurances > 0:
            for a in range(n_actions):
                dist[s][a] = sa_pairs[s][a]/total_occurances

    return dist


# calc state frequencies
def get_state_freqs(env, trajs, pol, T):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    mi = np.ones(n_states)

    # transition probs p(s'|s,a)
    trans_probs = get_transition_probs(env)
    #ref_probs = 

    
    for t in range(T):

        mi_new = mi

        for s_n in range(n_states):
            new_val = 0

            for s in range(n_states):

                for a in range(n_actions):

                    new_val += probs[s][a][s_n]*mi[s]*pol[s]

            mi_new[s_n] = new_val


    s_freq = (1/T) * mi_new

    return s_freq



def main():
    env = gym.make('FrozenLake-v0')
    env.render()
    print("")
    env.seed(0)
    np.random.seed(0)
    expertpolicy = [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    trajs = generate_demonstrations(env, expertpolicy, 0.1, 20)  # list of trajectories
    print("one trajectory is a list with (state, action) pairs:")
    print (trajs[0])

    ### (a) naive policy
    max_pairs = get_naive_pol(env, trajs)
    naive_pol = list(max_pairs.values())

    print("naive:  " + str(naive_pol))


    ### (b) state visiting freqs
    T = max([len(traj) for traj in trajs])
    #state_freqs = get_state_freqs(env, trajs, naive_pol, T)
    #probs = get_state_probs(env, trajs)
    dit = get_reference_dist(env, trajs)


    # get pol
    #opt_pol = value_iteration(env, rewards)
    print("blub")




if __name__ == "__main__":
    main()