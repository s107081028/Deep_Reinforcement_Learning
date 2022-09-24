import numpy as np
import copy
V = np.zeros((4, 4))
action_space = [[0, 1], [-1, 0], [0, -1], [1, 0]]                    # {right, up, left, down}
delta = 1
gamma = 0.1
theta = 0.0001
iter = 0
while(delta >= theta):
    iter += 1
    delta = 0
    V_copy = copy.deepcopy(V)
    for i in range(4):
        for j in range(4):
            rewards = 0
            for action in action_space:
                reward = 0
                action_i = 0
                action_j = 0
                # terminate
                if (i, j) == (0, 0) or (i, j) == (3, 3):
                    reward = 0
                # out of map -> stay initial state
                elif (i + action[0] == 4) or (i + action[0] == -1) or (j + action[1] == 4) or (j + action[1] == -1):
                    reward = -1
                # do action
                else:
                    reward = -1
                    action_i = action[0]
                    action_j = action[1]
                rewards += (1/len(action_space)) * (reward + gamma * V_copy[i + action_i, j + action_j])
            delta = max(delta, abs(V_copy[i, j] - rewards))
            V[i, j] = rewards
print(iter)
V = np.round(V, decimals=2)
np.save("107081028_hw1_1_data_gamma_0.1", V)
output = ''
for i in range(4):
    for j in range(4):
        output += str(V[i][j])
        if j != 4:
            output += " "
path = "107081028_hw1_1_data_gamma_" + str(gamma)
f = open(path, 'w')
f.write(output)
f.close()
print(V)