from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
action_space = [i for i in range(9)]
tictactoe = [-1, 0, 1]
states = []
for i in range(19683):
    count = i
    state = []
    for j in range(9):
        state.append(tictactoe[int(count % 3)])
        count /= 3
    states.append(state)

cross_reward = 1
circle_reward = 0

#%%
class QL():
    def __init__(self, action_space, states):
        self.action_space = action_space
        self.states = states
        self.table = np.ones((len(states))) * 0.5
        self.epsilon = 0.4
        
    def next_state_remain(self, state, actions, O_X):
        state_actions = []
        for action in actions:
            current = states[state].copy()
            current[action] = O_X
            state_index = self.states.index(current)
            state_actions.append((state_index, action))
        return state_actions

    def choose_action(self, state, actions, O_X):
        if(np.random.uniform() >= self.epsilon):
            action = np.random.choice(actions)
            current = states[state].copy()
            current[action] = O_X
            state_index = self.states.index(current)
            return state_index, self.table[state_index], action
        else:
            state_actions = self.next_state_remain(state, actions, O_X)
            if O_X == 1:
                value = -sys.float_info.max
                chosen_action = -1
                for (indices, action) in state_actions:
                    if self.table[indices] > value:
                        value = self.table[indices]
                        state_index = indices
                        chosen_action = action
                return state_index, value, chosen_action
                
            else:
                value = sys.float_info.max
                chosen_action = -1
                for (indices, action) in state_actions:
                    if self.table[indices] < value:
                        value = self.table[indices]
                        state_index = indices
                        chosen_action = action
                return state_index, value, chosen_action

    def learn(self, state_index, next_state_index, action):
        value = self.table[state_index]
        next = self.table[next_state_index]
        self.table[state_index] += 0.01 * (next - value)
#           next = reward + 0.9 * self.table[next_state, :].max()
        #print(np.reshape(self.states[state_index], (3,3)))
        #print(action)
        #print(f"{value} to {self.table[state_index]}")
            
    def terminate(self, state, reward):
        self.table[state] = reward
    
    def test_epsilon(self):
        self.epsilon = 1.0
        
    def train_epsilon(self):
        self.epsilon = 0.8
    

Model1 = QL(action_space, states)
win_cross = 0
win_circle = 0
tie = 0

#%%
def rewarding(state):
    ending_board = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), \
        (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for(i, j, k) in ending_board:
        if(sum([state[i], state[j], state[k]]) == -3):
            return circle_reward
        if(sum([state[i], state[j], state[k]]) == 3):
            return cross_reward
    if 0 not in state:
        return 0.5
    return -1

def make_action(state_index, O_X):
    reward = rewarding(states[state_index])
    terminate = False
    if reward == 1 or reward == 0 or reward == 0.5:
        terminate = True
    return terminate, reward

def can_make_action(state):
    can_index = []
    for i in range(len(state)):
        if state[i] == 0:
            can_index.append(i)
    return can_index

rewardX = []
rewardO = []
for _ in range(300000):
    state = [0 for i in range(9)]
    terminate = False
    action_list = [i for i in range(9)]
    state_index = states.index(state)
    O_X = random.choice([-1, 1])
    while terminate == False:
        next_state_index, value, action = Model1.choose_action(state_index, action_list, O_X)
        action_list.remove(action)
        terminate, reward = make_action(next_state_index, O_X)
        if terminate:
            Model1.terminate(next_state_index, reward)
            if reward == 1:
                win_cross += 1
            elif reward == 0.5:
                tie += 1
            else:
                win_circle += 1
        Model1.learn(state_index, next_state_index, action)
        state_index = next_state_index
        O_X *= -1
    if _ % 1000 == 0:
        print(_)
        print("win of cross: " + str(win_cross) + "\nwin of circle: " + str(win_circle) + "\ntie: " + str(tie))
        print(states[state_index])
    if _ == 200000:
        Model1.train_epsilon()
        
plt.figure()
plt.plot(rewardX)
plt.show
plt.figure()
plt.plot(rewardO)
plt.show

        
#%%
np.save("107081028_hw1_3_data_readonly.npy", Model1.table)

#%%
#Model1.test_epsilon()
Model1.epsilon = 1.0
for _ in range(10):
    state = [0 for i in range(9)]
    terminate = False
    action_list = [i for i in range(9)]
    state_index = states.index(state)
    O_X = -1
    result = ["O wins", "X wins", "Draw"]
    while terminate == False:
        terminate, reward = make_action(state_index, O_X)
        if terminate:
            if reward == 1:
                index = 1
            elif reward == 0.5:
                index = 2
            else:
                index = 0
            break
        next_state_index, value, action = Model1.choose_action(state_index, action_list, O_X)
        action_list.remove(action)
        state_index = next_state_index
        O_X *= -1        
    print(result[index])
    print(states[state_index])

