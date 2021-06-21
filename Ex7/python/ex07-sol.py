import gym
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict, deque

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break

def create_tiling_grid(low, high, bins=(20, 20), offsets=(0.0, 0.0)): # bins=(20) for 20 bin for 2 dim
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]
    return grid

def create_tilings(low, high, tiling_specs):
    return [create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs]

def discretize(sample, grid):
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension

def tile_encode(sample, tilings, flatten=False):
    encoded_sample = [discretize(sample, grid) for grid in tilings]
    return np.concatenate(encoded_sample) if flatten else encoded_sample

class QTable:
    """Simple Q-table."""

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        #print("QTable(): size =", self.q_table.shape)
    
    def scale_all(self, value):
        self.q_table *= value


class TiledQTable:
    """Composite Q-table with an internal tile coding scheme."""
    
    def __init__(self, low, high, tiling_specs, action_size):

        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits)+1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = [QTable(state_size, self.action_size) for state_size in self.state_sizes]
        #print("TiledQTable(): no. of internal tables = ", len(self.q_tables))
    
    def get(self, state, action):
        encoded_state = tile_encode(state, self.tilings)
        
        value = 0.0
        for idx, q_table in zip(encoded_state, self.q_tables):
            value += q_table.q_table[tuple(idx + (action,))]
        value /= len(self.q_tables)
        return value
    
    def update(self, state, action, value, alpha=0.1):
        encoded_state = tile_encode(state, self.tilings)
        
        for idx, q_table in zip(encoded_state, self.q_tables):
            value_ = q_table.q_table[tuple(idx + (action,))]  # current value
            q_table.q_table[tuple(idx + (action,))] = alpha * value + (1.0 - alpha) * value_
    
    def scale_all(self, value):
        for tb in self.q_tables:
            tb.scale_all(value)

def eps_greedy_policy(Q, state, eps=0.2):
    action_values = [Q.get(state, a) for a in range(Q.action_size)]
    greedy_action = np.argmax(action_values)
    action_prob = np.ones(Q.action_size) * eps / Q.action_size
    action_prob[greedy_action] += (1 - eps)
    return action_prob

def sarsa_lambda(env, low, high, tiling_specs, alpha=0.1, gamma=0.9, Lambda=0.9, eps_decay=0.9955, min_eps=0.01, num_eps=int(1e4)):
    nA = env.action_space.n # action set size of env
    num_steps_average = []
    step_window = deque(maxlen=100)
    num_cummulative_success = []
    score_window = deque(maxlen=100)
    num_success = 0
    eps = 1.0
    
    # initialize action-value func and e-greedy policy
    Q = TiledQTable(low, high, tiling_specs, nA)
    
    for episode in range(1, num_eps + 1):
        state = env.reset()
        action = np.random.choice(nA, p=eps_greedy_policy(Q, state, eps))
        
        done = False
        eligibility_traces = TiledQTable(low, high, tiling_specs, nA) # for tracking return of visited states
        t = 0
        score = 0
        while not done:         
            # update routine of discount factor 
            eligibility_traces.scale_all(gamma * Lambda) 
            current = eligibility_traces.get(state, action)
            eligibility_traces.update(state, action, current + 1.0, 1.0) 
            
            next_state, reward, done, _ = env.step(action)
            score += reward

            if done:
                target = reward
            else:
                next_action = np.random.choice(nA, p=eps_greedy_policy(Q, next_state, eps))
                target = reward + gamma * Q.get(next_state, next_action)

            #update action-value
            Q.update(state, action, target, alpha * eligibility_traces.get(state, action)) 
            
            state, action = next_state, next_action
            t += 1
        if t < 199:
            num_success += 1 
        step_window.append(t)
        num_steps_average.append(np.mean(step_window))
        num_cummulative_success.append(num_success)
        score_window.append(score)

        eps = max(min_eps, eps * eps_decay)
        if episode % 100 == 0:
            print("Running mean score: %s" % np.mean(score_window))
    return Q, num_steps_average, num_cummulative_success

def main():
    
    env = gym.make('MountainCar-v0')
    env.reset()
    one_tiling = [((20, 20), (0.0, 0.0))]
    tiling_specs = [((20, 20), (-0.05, -0.005)),
                    ((20, 20), (0.0, 0.0)),
                    ((20, 20), (0.05, 0.005))]
    low, high = env.observation_space.low, env.observation_space.high
    num_eps = int(5e3)
    
    steps = []
    successes = []
    for _ in range(2):
        _, num_steps_average, num_cummulative_success = sarsa_lambda(env, low, high, one_tiling, num_eps=num_eps)
        steps.append(num_steps_average)
        successes.append(num_cummulative_success)

    steps = np.array(steps)
    steps = steps.mean(axis=0)
    successes = np.array(successes)
    successes = successes.mean(axis=0)

    x = np.arange(1, num_eps + 1)
    plt.subplot(2, 1, 1)
    plt.plot(x, steps)
    plt.title('Average steps over episodes')
    plt.ylabel('Steps')

    plt.subplot(2, 1, 2)
    plt.plot(x, successes)
    plt.title('Average cummulative successes over episodes')
    plt.ylabel('Success')

    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
