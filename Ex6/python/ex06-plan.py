import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent  # parent of this node
        self.action = action  # action leading from parent to this node
        self.children = []
        self.sum_value = 0.  # sum of values observed for this node, use sum_value/visits for the mean
        self.visits = 0


def rollout(env, maxsteps=100):
    """ Random policy for rollouts """
    G = 0
    for i in range(maxsteps):
        action = env.action_space.sample()
        _, reward, terminal, _ = env.step(action)
        G += reward
        if terminal:
            return G
    return G


def mcts(env, root, maxiter=500):
    """ TODO: Use this function as a starting point for implementing Monte Carlo Tree Search
    """

    # this is an example of how to add nodes to the root for all possible actions:
    #root.children = [Node(root, a) for a in range(env.action_space.n)]

    for i in range(maxiter):
        state = copy.deepcopy(env)
        G = 0.
        eps = 0.1
        terminal = False
        longest_path = 0
        

        ## TODO: traverse the tree using an epsilon greedy tree policy

        # start with given root of tree
        node = root

        # traverse tree until leaf is reached
        while len(node.children) > 0:

            # rndm action
            if random.random() < eps:

                # choose random child
                node = random.choice(node.children)

            # greedy action
            else:

                # get values from childs
                child_vals = []
                for c in node.children:
                    #child_vals.append(c.sum_value)
                    if c.visits > 0:
                        child_vals.append(c.sum_value/c.visits)
                    else:
                        child_vals.append(c.sum_value)

                # choose child with largest val
                node = node.children[np.argmax(child_vals)]  

                # do step
                _, reward, terminal, _ = state.step(node.action)
                G += reward       
        

        ## TODO: Expansion of tree

        if len(node.children) == 0  and terminal == False:
            node.children = [Node(node, a) for a in range(state.action_space.n)]
            node = random.choice(node.children)
            _, reward, terminal, _ = state.step(node.action)
            G += reward   

        # This performs a rollout (Simulation):
        if not terminal:
            G += rollout(state)

        # TODO: update all visited nodes in the tree
        # This updates values for the current node:
        #node.visits += 1
        #node.sum_value += G

        depth = 0

        while node.parent != None:

            # update vals
            node.visits += 1
            node.sum_value += G

            # update depth
            depth += 1

            # move to next parent
            node = node.parent

        if depth > longest_path:
            longest_path = depth


    return longest_path




def main():
    env = gym.make("Taxi-v3")
    env.seed(0)  # use seed to make results better comparable
    # run the algorithm 10 times:
    rewards = []
    paths = []
    longest_paths = []
    for i in range(10):
        env.reset()
        terminal = False
        root = Node()  # Initialize empty tree
        sum_reward = 0.
        while not terminal:
            env.render()
            longest_path = mcts(env, root)  # expand tree from root node using mcts
            paths.append(longest_path)
            values = [c.sum_value/c.visits for c in root.children]  # calculate values for child actions
            bestchild = root.children[np.argmax(values)]  # select the best child
            _, reward, terminal, _ = env.step(bestchild.action) # perform action for child
            root = bestchild  # use the best child as next root
            root.parent = None
            sum_reward += reward
        rewards.append(sum_reward)
        longest_paths.append(max(paths))
        paths = []
        print("finished run " + str(i+1) + " with reward: " + str(sum_reward))
    print("mean reward: ", np.mean(rewards))
    print("rewards: " + str(rewards))

    # plot figures
    fig = plt.figure(1)
    plt.plot(rewards, label="rewards")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("reward")

    fig2 = plt.figure(2)
    plt.plot(longest_paths, label="longest paths")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("path length")

    plt.show()

if __name__ == "__main__":
    main()