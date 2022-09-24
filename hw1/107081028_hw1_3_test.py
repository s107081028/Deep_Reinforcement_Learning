# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 00:15:50 2022

@author: stanl
"""
import numpy as np
import sys
# action_space = [i for i in range(9)]
# tictactoe = [-1, 0, 1]
# states = []
# for i in range(19683):
#     count = i
#     state = []
#     for j in range(9):
#         state.append(tictactoe[int(count % 3)])
#         count /= 3
#     states.append(state)

def can_make_action(state):
    can_index = []
    for i in range(len(state)):
        if state[i] == 0:
            can_index.append(i)
    return can_index

def next_state_remain(states, state, actions, O_X):
    state_actions = []
    for action in actions:
        current = states[state].copy()
        current[action] = O_X
        state_index = states.index(current)
        state_actions.append((state_index, action))
    return state_actions

def choose_action(table, states, state, actions, O_X):
    state_actions = next_state_remain(states, state, actions, O_X)
    if O_X == 1:
        value = -sys.float_info.max
        chosen_action = -1
        for (indices, action) in state_actions:
            #print(table[indices])
            if table[indices] > value:
                value = table[indices]
                state_index = indices
                chosen_action = action
        return state_index, value, chosen_action
    else:
        value = sys.float_info.max
        chosen_action = -1
        for (indices, action) in state_actions:
            #print(table[indices])
            if table[indices] < value:
                value = table[indices]
                state_index = indices
                chosen_action = action
        return state_index, value, chosen_action
        
# with open("hw1-3_sample_input") as file:
#     for lines in file:
#         line = lines.split()
#         player = int(line[0])
#         model = np.load("107081028_model_only4.npy")
#         state = [int(i) for i in line[1:]]
#         state_index = states.index(state)
#         capable_action = can_make_action(state)
#         next_state_index, value, action = choose_action(model, states, state_index, capable_action, player)
#         for i in range(len(line)):
#             if int(line[i]) == -1:
#                 line[i] = "O"
#             elif int(line[i]) == 0:
#                 line[i] = "?"
#             elif int(line[i]) == 1:
#                 line[i] = "X"
#         print(line[0])
#         for i in range(3):
#             print(f"{line[1 + i * 3]} {line[1 + i * 3 + 1]} {line[1 + i * 3 + 2]}")
#         print(f"{(action % 3)} {int((action / 3))}\n")
#         f = open("hw1-3_output", "a")
#         f.write(f"{(action % 3)} {int((action / 3))}\n")
#         f.close()

#%%
if __name__ =='__main__':
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
    model = np.load("./107081028_hw1_3_data.npy")
    try:
        while True:
            s = input()
            line = s.split()
            player = int(line[0])
            state = [int(i) for i in line[1:]]
            state_index = states.index(state)
            capable_action = can_make_action(state)
            next_state_index, value, action = choose_action(model, states, state_index, capable_action, player)
            for i in range(len(line)):
                if int(line[i]) == -1:
                    line[i] = "O"
                elif int(line[i]) == 0:
                    line[i] = "?"
                elif int(line[i]) == 1:
                    line[i] = "X"
#            print(line[0])
#            for i in range(3):
#                print(f"{line[1 + i * 3]} {line[1 + i * 3 + 1]} {line[1 + i * 3 + 2]}")
            print(f"{(action % 3)} {int((action / 3))}")
#            f = open("hw1-3_output", "a")
#            f.write(f"{(action % 3)} {int((action / 3))}\n")
#            f.close()
    except EOFError:
        pass