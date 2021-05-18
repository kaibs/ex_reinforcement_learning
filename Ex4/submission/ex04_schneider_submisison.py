import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


# 
def eval_res(vals, x, y, useable_ace):

    res = np.zeros(x.shape)

    for i in range(y.shape[0]):
        for j in range(x.shape[1]):
            res[i,j] = vals[(int(x[i,j]),int(y[i,j]), useable_ace)]

    return res


# run first-visit MC
def first_visit_mc(episodes):

    env = gym.make('Blackjack-v0')
    obs = env.reset()
    V = {}
    R = {}

    # init cases
    for player_sum in range(2, 22):
        for dealer_card in range(1, 11):
            for useable_ace in [False, True]:
                R[(player_sum, dealer_card, useable_ace)] = []
                V[(player_sum, dealer_card, useable_ace)] = 0

    # go over episodes
    for e in range(episodes):

        # do episode
        rew = 0
        eps = []
        done = False
        obs = env.reset()
        while not done:
            eps.append((obs[0], obs[1], obs[2]))
            obs, rew, done, _ = env.step(0 if obs[0] >= 0 else 1)

        # eval episode
        for s in eps:
            
            R[(s[0], s[1], s[2])].append(rew)
            V[(s[0], s[1], s[2])] = sum(R[(s[0], s[1], s[2])])/len(R[(s[0], s[1], s[2])])


        if e%1000 == 0:
            print(str(e) + " episodes done")
        
    return V


# plot ES MC pols
def plot_pol(pol, eps):

    player = list(range(12,22))
    dealer = list(range(1,11))

    res_ace = np.zeros((10, 10))
    res_nace = np.zeros((10, 10)) 

    for p in range(len(player)): 
        for d in range(len(dealer)):
            res_ace[p,d] = pol[(player[p], dealer[d], True)]
            res_nace[p,d] = pol[(player[p], dealer[d], False)]

    fig, (ax0, ax1) = plt.subplots(1, 2)

    plt.rcParams.update({'font.size': 22})

    plt.suptitle(str(eps/1000)+  "k eps")

    c = ax0.pcolor(res_ace, vmin=0, vmax=1)
    ax0.set_title('useable ace')
    ax0.set_xlabel('playersum')
    ax0.set_ylabel('dealercard')
    ax0.set_xticks(list(range(0, 11)))
    ax0.set_yticks(list(range(0, 11)))
    ax0.set_xticklabels(player, fontsize=16)
    ax0.set_yticklabels(dealer, fontsize=16)
    fig.colorbar(c, ax=ax0)
    
    c = ax1.pcolor(res_nace, vmin=0, vmax=1)
    ax1.set_title('unuseable ace')
    ax1.set_xlabel('playersum')
    ax1.set_ylabel('dealercard')
    ax1.set_xticks(list(range(0, 11)))
    ax1.set_yticks(list(range(0, 11)))
    ax1.set_xticklabels(player, fontsize=16)
    ax1.set_yticklabels(dealer, fontsize=16)
    fig.colorbar(c, ax=ax1)
    
    plt.show()



# run exploring starts MC
def every_visit_mc(episodes):

    env = gym.make('Blackjack-v0')
    obs = env.reset()
    Q = {}
    R = {}
    pol = {}

    # init cases
    for player_sum in range(2, 22):
        for dealer_card in range(1, 11):
            for useable_ace in [False, True]:
                for action in [0,1]:
                    R[(player_sum, dealer_card, useable_ace, action)] = []
                    Q[(player_sum, dealer_card, useable_ace, action)] = 0
                    pol[(player_sum, dealer_card, useable_ace)] = np.random.randint(2)


    # loop over episodes
    for e in range(episodes):

        # do episode
        eps = []
        rew = []
        done = False
        obs = env.reset()

        # init first state of eps as rndm
        act = np.random.randint(2)
        eps.append((obs[0], obs[1], obs[2], act))
        obs, rew, done, _ = env.step(act)

        # rest with pol
        while not done:
            act = pol[(obs[0], obs[1], obs[2])]
            eps.append((obs[0], obs[1], obs[2], act))
            obs, rew, done, _ = env.step(act)

        
        for sa in eps:
            R[(sa[0], sa[1], sa[2], sa[3])].append(rew)
            Q[(sa[0], sa[1], sa[2], sa[3])] = sum(R[(sa[0], sa[1], sa[2], sa[3])])/len(R[(sa[0], sa[1], sa[2], sa[3])])

            if Q[(sa[0], sa[1], sa[2], 1)] > Q[(sa[0], sa[1], sa[2], 0)]: 
                opt_act = 1
            else:
                opt_act = 0

            pol[(sa[0], sa[1], sa[2])] = opt_act
                              

        if e%10000 == 0:

            if e%100000 == 0:
                print("opt_pol @" + str(e/1000) + "k: " + str(pol))
                plot_pol(pol, e)
            else:
                print(str(e/1000) + "k episodes done")
    


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    while not done:
        print("observation:", obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)
        print("reward:", reward)
        print("")


    # MC ES 1e6 eps:
    es_vals = every_visit_mc(1000001)

    
    # fist: 10k eps
    vals = first_visit_mc(10000)

    x,y  = np.meshgrid(np.array(range(12,22)), np.array(range(1,11)))

    fig = plt.figure()
    plt.suptitle("10k eps")
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_wireframe(x,y, eval_res(vals,x,y, True))
    ax.set_title('useable ace')
    ax.set_xlabel('playersum')
    ax.set_ylabel('dealercard')
    ax.set_zlabel('average reward')
    
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_wireframe(x,y, eval_res(vals,x,y, False))
    ax.set_title('unuseable ace')
    ax.set_xlabel('playersum')
    ax.set_ylabel('dealercard')
    ax.set_zlabel('average reward')
    
    plt.show()


    # fist: 500k eps
    vals = first_visit_mc(500000)

    fig2 = plt.figure()
    plt.suptitle("500k eps")
    ax = fig2.add_subplot(121, projection='3d')
    ax.plot_wireframe(x,y, eval_res(vals,x,y, True))
    ax.set_title('useable ace')
    ax.set_xlabel('playersum')
    ax.set_ylabel('dealercard')
    ax.set_zlabel('average reward')
    
    ax = fig2.add_subplot(122, projection='3d')
    ax.plot_wireframe(x,y, eval_res(vals,x,y, False))
    ax.set_title('unuseable ace')
    ax.set_xlabel('playersum')
    ax.set_ylabel('dealercard')
    ax.set_zlabel('average reward')
    
    plt.show()
  
    print("finished")


if __name__ == "__main__":
    main()