import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import glob
import math
import base64
import gym
import slimevolleygym
from collections import deque
env = gym.make("SlimeVolleyNoFrameskip-v0")
num_state_feats = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")
class DuelingDQN(nn.Module):
  def __init__(self, num_actions):
    super(DuelingDQN, self).__init__()
    self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    std = math.sqrt(2.0 / (4 * 84 * 168))
    nn.init.normal_(self.conv1.weight, mean=0.0, std=std)
    self.conv1.bias.data.fill_(0.0)

    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    std = math.sqrt(2.0 / (32 * 3 * 8 * 8))
    nn.init.normal_(self.conv2.weight, mean=0.0, std=std)
    self.conv2.bias.data.fill_(0.0)

    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    std = math.sqrt(2.0 / (64 * 32 * 4 * 4))
    nn.init.normal_(self.conv3.weight, mean=0.0, std=std)
    self.conv3.bias.data.fill_(0.0)

    self.fc1 = nn.Linear(7616, 512)
    std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
    nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
    self.fc1.bias.data.fill_(0.0)
    self.V = nn.Linear(512, 1)
    self.A = nn.Linear(512, num_actions)


  def forward(self, states):
    x = F.relu(self.conv1(states))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.fc1(x.view(x.size(0), -1)))
    V = self.V(x)
    A = self.A(x)
    Q = V + (A - A.mean(dim=1, keepdim=True))
    return Q

class Agent(object):
    def __init__(self):
        self.nn =  DuelingDQN(num_actions).to(device)
        self.nn.load_state_dict(torch.load("107081028_hw2_data.pt"))
        self.frameQue = deque([], maxlen=4)

    def act(self, observation):
        observation = np.array(np.dot(observation, [0.299, 0.587, 0.114]), dtype=np.float32)
        if len(self.frameQue) == 0:
            for i in range(4):
              self.frameQue.append(observation)
              
        else:
            self.frameQue.append(observation)
        state = list(self.frameQue)
        state = np.array(state)
        state = torch.FloatTensor(np.expand_dims(state / 255., axis=0)).to(device)
        action = np.argmax(self.nn(state).cpu().data.numpy())
        return action