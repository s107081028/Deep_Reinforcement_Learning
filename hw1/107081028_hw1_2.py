# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:08:17 2022

@author: stanl
"""

import numpy as np
import matplotlib.pyplot as plt
alpha = 0.5
gamma = 1.0
reward_plot = list()
qlr_table = np.zeros((4, 12, 4))

def q_learning():
    reward_plot = list()
    qlr_table = np.zeros((4, 12, 4))
    for i in range(500):
        terminate = False
        current_environment = np.zeros((4, 12))
        current_environment[3][0] = 1
        current_row = 3
        current_col = 0
        reward_accumulatived = 0
        step = 1
        while(terminate == False):
            step += 1
            rng = np.random.random()
            action = 3
            if(rng < 0.05):
              action = np.random.choice(4)
            else:
               action = np.argmax(qlr_table[current_row, current_col, :])
            # action = np.argmax(qlr_table[current_row, current_col, :])
            next_row = current_row
            next_col = current_col
            if(action == 0 and current_row > 0):
                next_row = current_row - 1
            elif(action == 1 and current_col > 0):
                next_col = current_col - 1
            elif(action == 2 and current_col < 11):
                next_col = current_col + 1
            elif(action == 3 and current_row < 3):
                next_row = current_row + 1
            current_environment[next_row][next_col] = step
            max_value = np.max(qlr_table[next_row, next_col, :])
            reward = -1
            if(next_row == 3 and next_col == 11):
                terminate = True
                reward = 0
            elif(next_row == 2 and next_col > 0 and next_col < 11):
                terminate = True
                reward = -100
            reward_accumulatived += reward
            next_qlr_value = qlr_table[current_row, current_col, action] + \
                alpha * (reward + gamma * max_value - qlr_table[current_row, current_col, action])
            qlr_table[current_row, current_col, action] = next_qlr_value
            current_row = next_row
            current_col = next_col
        reward_plot.append(reward_accumulatived)
        if(i % 100 == 0):
            print(current_environment)
    plt.figure()
    plt.plot(reward_plot)
    plt.ylim(-100, 0)
    plt.show
    return reward_plot

def sarsa():
    reward_plot = list()
    sarsa_table = np.zeros((4, 12, 4))
    for i in range(500):
        terminate = False
        current_environment = np.zeros((4, 12))
        current_row = 3
        current_col = 0
        reward_accumulatived = 0
        action = np.argmax(sarsa_table[current_row, current_col, :])
        step = 1
        while(terminate == False):
            step += 1
            next_row = current_row
            next_col = current_col
            if(action == 0 and current_row > 0):
                next_row = current_row - 1
            if(action == 1 and current_col > 0):
                next_col = current_col - 1
            if(action == 2 and current_col < 11):
                next_col = current_col + 1
            if(action == 3 and current_row < 3):
                next_row = current_row + 1
            current_environment[current_row][current_col] = step
            reward = -1
            if(next_row == 3 and next_col == 11):
                terminate = True
                reward = 0
            elif(next_row == 2 and next_col > 0 and next_col < 11):
                terminate = True
                reward = -100
            reward_accumulatived += reward
            rng = np.random.random()
            next_action = 3
            if(rng < 0.05):
               next_action = np.random.choice(4)
            else:
               next_action = np.argmax(sarsa_table[next_row, next_col, :])
            # next_action = np.argmax(sarsa_table[next_row, next_col, :])
            value = sarsa_table[next_row, next_col, next_action]
            next_sarsa_value = sarsa_table[current_row, current_col, action] + \
                alpha * (reward + gamma * value - sarsa_table[current_row, current_col, action])
            sarsa_table[current_row, current_col, action] = next_sarsa_value
            current_row = next_row
            current_col = next_col
            action = next_action
        reward_plot.append(reward_accumulatived)
        if(i % 100 == 0):
            print(current_environment)
    plt.figure()
    plt.plot(reward_plot)
    plt.ylim(-100, 0)
    plt.show
    return reward_plot

qlr_plot = q_learning()
sarsa_plot = sarsa()

qlr_norm = []
qlr_mean = np.array(qlr_plot).mean()
qlr_std = np.array(qlr_plot).std()
tick = 0
temp = 0
for rewards in qlr_plot:
    tick += 1
    temp += rewards
    if(tick == 20):
        reward_norm = (temp - qlr_mean) / qlr_std
        qlr_norm.append(reward_norm)
        tick = 0
        temp = 0

sarsa_norm = []
sarsa_mean = np.array(sarsa_plot).mean()
sarsa_std = np.array(sarsa_plot).std()
tick = 0
temp = 0
for rewards in sarsa_plot:
    tick += 1
    temp += rewards
    if(tick == 20):
        reward_norm = (temp - sarsa_mean) / sarsa_std
        sarsa_norm.append(reward_norm)
        tick = 0
        temp = 0

plt.figure()
plt.plot(qlr_norm, label = "q_learning")
plt.plot(sarsa_norm, label = "sarsa")
plt.show