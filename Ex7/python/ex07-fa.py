import gym
import numpy as np
import random
import matplotlib.pyplot as plt


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


# create q table
def create_qtab(nbr_states, nbr_actions):
    
    return np.zeros((nbr_states, nbr_actions))


# select greedy action
def greedy_pol(Q, s):

    act_arr = [a for a in Q[s, :]]
    a = np.argmax(act_arr)

    return a


# choose random action
def rndm_pol(Q, env):

    a = env.action_space.sample()

    return a


def q_episodes(env, Q):

    alpha = 0.1 # stepsize / learning rate
    epsilon = 0.2 # explorativity
    gamma = 0.8 # discount factor

    # go over eps
    cntr = 0
    for eps in range(200):

        # reset env
        env.reset()

        # loop over states
        for s in range(np.shape(Q)[0]):


            # explore
            if random.random() < epsilon:
                a = env.action_space.sample()

            # exploit
            else:
                a = greedy_pol(Q, s)

            # do step
            s_new, reward, done, _ = env.step(a)

            s_new[0]

            # update qtab
            Q[s, a] = Q[s, a] + alpha * (reward + gamma * np.max(Q[s_new, :]) - Q[s, a])
            

            # break condition if goal reached faster
            if s_new >= 0.5 or done == True:
                break

            
            if (cntr % 50) == 0:

                print("observation: ", s_new)
                print("reward: ", reward)




def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    #random_episode(env)

    Q = create_qtab(20, 3)
    q_episodes(env, Q)



    env.close()


if __name__ == "__main__":
    main()