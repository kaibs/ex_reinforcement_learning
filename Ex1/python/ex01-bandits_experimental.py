import numpy as np
import matplotlib.pyplot as plt
import random
import time


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    for i in possible_arms:
        reward_a = bandit.play_arm(i)
        rewards[i] += reward_a
        n_plays += 1
        Q[i] = rewards[i]

    # Main loop
    while bandit.total_played < timesteps:
        # This example shows how to play a random arm:
        #a = random.choice(possible_arms)
        #reward_for_a = bandit.play_arm(a)

        # TODO: instead do greedy action selection
        a = np.argmax(Q)
        reward_a = bandit.play_arm(a)

        # TODO: update the variables (rewards, n_plays, Q) for the selected arm
        rewards[a] += reward_a
        n_plays[a] += 1
        Q[a] = rewards[a]/n_plays[a]


def epsilon_greedy(bandit, timesteps):
    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)
    eps = 0.1

    # init vars
    for i in possible_arms:
        reward_a = bandit.play_arm(i)
        rewards[i] += reward_a
        n_plays += 1
        Q[i] = rewards[i]
    
    
    while bandit.total_played < timesteps:

        # random action
        if random.random() < eps:
            a = random.randint(0,(bandit.n_arms-1))

        # greedy action
        else:
            a = np.argmax(Q)

        # play arm & update vars
        reward_a = bandit.play_arm(a)
        rewards[a] += reward_a
        n_plays[a] += 1
        Q[a] = rewards[a]/n_plays[a]


def variable_greedy(bandit, timesteps):
    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    eps = 0.5 # init with eps way higher than before
    eps_add = float(0.4/timesteps)

    # init vars
    for i in possible_arms:
        reward_a = bandit.play_arm(i)
        rewards[i] += reward_a
        n_plays += 1
        Q[i] = rewards[i]


    
    
    while bandit.total_played < timesteps:

        # random action
        if random.random() < (eps - bandit.total_played*eps_add):
            a = random.randint(0,(bandit.n_arms-1))

        # greedy action
        else:
            a = np.argmax(Q)

        # play arm & update vars
        reward_a = bandit.play_arm(a)
        rewards[a] += reward_a
        n_plays[a] += 1
        Q[a] = rewards[a]/n_plays[a]



def main():
    n_episodes = 15000  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1500
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)
    rewards_vgreedy = np.zeros(n_timesteps)

    start = time.time()

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        variable_greedy(b, n_timesteps)
        rewards_vgreedy += b.rewards


    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    rewards_vgreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.plot(rewards_vgreedy, label="v-greedy")
    print("Total reward of variable greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_vgreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()

    print("total processing time: " + str(time.time() - start))


if __name__ == "__main__":
    main()
