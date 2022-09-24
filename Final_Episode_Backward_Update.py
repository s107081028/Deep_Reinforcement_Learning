# -*- coding: utf-8 -*-
"""EBU_pong.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19CJh_cRcld8t5NJ4H41xIl7lBpGyYo5Z
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install pyglet==1.5.7
!pip install gym==0.19.0
!pip install gym[atari]

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/DRL_final/

!ls

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gym
import os
from collections import deque
import pickle
from gym.wrappers import Monitor, AtariPreprocessing, FrameStack
from datetime import datetime
from math import ceil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = gym.make("PongDeterministic-v4")
env = AtariPreprocessing(env, frame_skip = 1)
env = FrameStack(env, num_stack=4)
numOfAct = env.action_space.n
PATH = "ckpt/checkpoints_EBU_pong_v4.pt"
print("numofact", numOfAct)

class DuelingDQN(nn.Module):
  def __init__(self, numOfAct):
    super(DuelingDQN, self).__init__()
    self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    self.bn1 = nn.BatchNorm2d(32)

    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.bn2 = nn.BatchNorm2d(64)

    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.fc1 = nn.Linear(3136, 512)
    self.LReLU1 = nn.LeakyReLU()

    self.fc2 = nn.Linear(3136, 512)
    self.LReLU2 = nn.LeakyReLU()

    self.V = nn.Linear(512, 1)
    self.A = nn.Linear(512, numOfAct)

  def forward(self, states):
    x = F.relu(self.bn1(self.conv1(states)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x1 = self.LReLU1(self.fc1(x.reshape(x.size(0), -1)))
    x2 = self.LReLU2(self.fc2(x.reshape(x.size(0), -1)))
    V = self.V(x1)
    A = self.A(x2)
    Q = V + (A - A.mean(dim=1, keepdim=True))
    return Q

main_net = DuelingDQN(numOfAct).to(device)
target_net = DuelingDQN(numOfAct).to(device)
optimizer = torch.optim.Adam(main_net.parameters(), lr=25e-5)
episodes = 0
loss = np.inf
epsilon =  1

x_axis_frames = []
y_axis_reward = []
x_axis_episode = []

last_100_ep_rewards, cur_frame = [], 0
if os.path.exists(PATH):
  checkpoint = torch.load(PATH)
  main_net.load_state_dict(checkpoint['model_state_dict'])
  target_net.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  episodes = checkpoint['episodes']
  loss = checkpoint['loss']
  epsilon = checkpoint['epsilon']
  cur_frame = checkpoint['cur_frame']
  x_axis_frames = checkpoint['exp_record']["x"]
  x_axis_episode = checkpoint['exp_record']["episode"]
  y_axis_reward = checkpoint['exp_record']["y"]
  last_100_ep_rewards = checkpoint['last_100_ep_rewards']
else:
  print("no path exist")

loss_fn = nn.SmoothL1Loss()

def select_act(state, epsilon):
  result = np.random.uniform()
  if result < epsilon:
    return env.action_space.sample() # Random action.
  else:
    main_net.eval()
    qs = main_net(state).cpu().data.numpy()
    main_net.train()
    return np.argmax(qs) # Greedy action for state.

class Buffer(object):
  def __init__(self, size, device):
    self._size = size
    self.buffer = []
    self.device = device
    self._next_idx = 0
    self.done = 1

  def add(self, state, action, reward, next_state, done):
    if self.done == 1:
        if self._next_idx >= self._size:
            self._next_idx = 0
            self.buffer[self._next_idx] = []
        elif len(self.buffer) < self._size:
            self.buffer.append([])
        else:
            self.buffer[self._next_idx] = []
        self.buffer[self._next_idx].append((state, action, reward, next_state, done))
        self._next_idx += 1
    else:
      self.buffer[self._next_idx - 1].append((state, action, reward, next_state, done))
    self.done = done

  def __len__(self):
    return len(self.buffer)

  def sample(self):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer))
    for i in self.buffer[idx]:
      state, action, reward, next_state, done = i
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = torch.as_tensor(np.array(states), device=self.device)
    states = states.type(torch.FloatTensor).to(device) / 255.
    # actions = torch.as_tensor(np.array(actions), device=self.device)
    # rewards = torch.as_tensor(np.array(rewards, dtype=np.float32),
    #                           device=self.device)
    next_states = torch.as_tensor(np.array(next_states), device=self.device)
    next_states = next_states.type(torch.FloatTensor).to(device) / 255.
    # dones = torch.as_tensor(np.array(dones, dtype=np.float32),
    #                         device=self.device)
    return states, actions, rewards, next_states, dones

def update(states, actions, y):
  actions = torch.as_tensor(np.array(actions), device=device)
  actions = actions.type(torch.LongTensor).to(device)
  main_net.eval()
  masked_qs = main_net(states).gather(1, actions.unsqueeze(dim=-1)).squeeze()
  main_net.train()
  y = torch.as_tensor(np.array(y)).to(device)

  loss = loss_fn(masked_qs, y)
  
  optimizer.zero_grad()
  loss.backward()
  for param in main_net.parameters():
      param.grad.data.clamp_(-1, 1)
  optimizer.step()
  return loss

import matplotlib.pyplot as plt
def plot_efficency(x, y, episode):
  if os.path.exists('DDQN_record_pong_v3.pickle'):
    with open('DDQN_record_pong_v3.pickle', 'rb') as f:
      new_dict = pickle.load(f)
    x2, y2 = new_dict['x'], new_dict['y']
    print(x2)
    print(y2)
    plt.plot(x2, y2, 'b-o', x, y, 'r')   # red line without marker
    plt.savefig(f'./plot/Pong_EBU_v2_{episode}.png', dpi=600)
    plt.show()

PATH = "ckpt/checkpoints_EBU_pong_v3_50.pt"
x_axis_frames = []
x_axis_episode = []
y_axis_reward = []
last_100_ep_rewards = []



buffer = Buffer(size=64, device=device)

for i in range(100):
  state = env.reset()
  ep_reward, done = 0, False
  while not done:
    state_np = np.array(state, dtype = np.float32)
    state_in = torch.as_tensor(np.expand_dims(state_np / 255., axis=0),
                               device=device)
    action = select_act(state_in, epsilon)
    next_state, reward, done, info = env.step(action)
    ep_reward += reward
    reward = np.sign(reward)
    buffer.add(state, action, reward, next_state, done)
    state = next_state
  print("epi", i , ep_reward)

with open('EBU_buffer_pong_v3.pickle', 'wb') as f:
    pickle.dump(buffer, f)

print(len(buffer))

num_episodes = 7000
batch_size = 64 
discount = 0.99 
buffer_size = 5000
B = 1
#writer = SummaryWriter()
# agent = Agent()
# buffer = Buffer(size=buffer_size, device=device)

# buffer = Buffer(size=buffer_size, device=device)

if os.path.exists('EBU_buffer_pong_v4.pickle'):
  with open('EBU_buffer_pong_v4.pickle', 'rb') as f:
      buffer = pickle.load(f)
  print("loaded buffer len:", len(buffer))
else:
  print("create buffer")
  buffer = Buffer(size=buffer_size, device=device)


# last_100_ep_rewards, cur_frame = [], 0
batchnum, batchcount = 0, 0
for episode in range(episodes, num_episodes+1):
  state = env.reset()
  ep_reward, done = 0, False
  print("episode:", episode, "cur_frame: ", cur_frame)
  while not done:
    state_np = np.array(state, dtype = np.float32)
    state_in = torch.as_tensor(np.expand_dims(state_np / 255., axis=0),
                               device=device)
    action = select_act(state_in, epsilon)
    next_state, reward, done, info = env.step(action)
    ep_reward += reward
    reward = np.sign(reward)
    if reward != 0:
      buffer.add(state, action, reward, next_state, True)
    else:
      buffer.add(state, action, reward, next_state, done)
    state = next_state
    cur_frame += 1
    if len(buffer) > 1 and batchnum == batchcount and cur_frame > 50000:
        states, actions, rewards, next_states, dones = buffer.sample()
        transition_length = len(actions)
        batchnum = ceil(transition_length / batch_size)
        batchcount = 1
        target_net.eval()
        _Q = target_net(next_states) #(sample_length, num_action)
        _Q[-1,:] = torch.as_tensor([0] * numOfAct)
        y = np.zeros(transition_length)
        y[-1] = rewards[-1]
        for i in range(transition_length - 2, -1, -1):
            _Q[i][actions[i + 1]] = B * y[i + 1] + (1 - B) * _Q[i][actions[i + 1]]
            y[i] = rewards[i] + discount * torch.max(_Q[i])
        target_net.train()
        loss = update(states[:batch_size], actions[:batch_size], y[:batch_size])
    elif len(buffer) > 1 and batchnum == batchcount + 1 and cur_frame > 50000:
        batchcount += 1
        loss = update(states[(batchcount-1) * batch_size:], actions[(batchcount-1) * batch_size:], y[(batchcount-1) * batch_size:])
    elif len(buffer) > 1 and cur_frame > 50000:
        batchcount += 1
        loss = update(states[(batchcount-1) * batch_size:batchcount * batch_size], actions[(batchcount-1) * batch_size:batchcount * batch_size], y[(batchcount-1) * batch_size:batchcount * batch_size])
    

    # Copy main_net weights to target_net.
    if cur_frame % 100 == 0 and cur_frame > 50000:
        if cur_frame % 10000 == 0:
          target_net.load_state_dict(main_net.state_dict())
        if cur_frame % 100 == 0:
            print(f'save frame reward: {np.mean(last_100_ep_rewards):.2f}')
            x_axis_frames.append(cur_frame)
            y_axis_reward.append(np.mean(last_100_ep_rewards))
            x_axis_episode.append(episode)
            if cur_frame % 3000 == 0 and epsilon > 0.05:   
              epsilon *= 0.99    

  if len(last_100_ep_rewards) == 100 and cur_frame > 50000:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)

  if episode % 10 == 0 and episode > 0 and cur_frame > 50000:
    
    print(f'Episode: {episode}/{num_episodes}, Epsilon: {epsilon:.3f}, '\
          f'Loss: {loss:.7f}, Return: {np.mean(last_100_ep_rewards):.2f}')
    print("save ckpt  ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    exp_record = {
        "x" : x_axis_frames,
        "y" : y_axis_reward,
        "episode" : x_axis_episode
    }
    torch.save({
            'episodes': episode,
            'model_state_dict': main_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'epsilon': epsilon,
            'exp_record': exp_record,
            'last_100_ep_rewards': last_100_ep_rewards,
            'cur_frame': cur_frame
            }, PATH)
    with open('EBU_record_pong_v4.pickle', 'wb') as f:
        pickle.dump(exp_record, f)
    plot_efficency(x_axis_frames, y_axis_reward, episode)
    with open('EBU_buffer_pong_v4.pickle', 'wb') as f:
        pickle.dump(buffer, f)

env.close()
