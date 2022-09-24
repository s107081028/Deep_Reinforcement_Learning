import gym,  math
import slimevolleygym
import numpy as np
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

env = gym.make("SlimeVolleyNoFrameskip-v0")
num_state_feats = env.observation_space.shape
num_actions = env.action_space.n
print(num_state_feats)

#%%
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

#%%
main_nn = DuelingDQN(num_actions).to(device)
target_nn = DuelingDQN(num_actions).to(device)

optimizer = torch.optim.Adam(main_nn.parameters(), lr=1e-5)
loss_fn = nn.SmoothL1Loss()

#%%
def select_epsilon_greedy_action(state, epsilon):
  result = np.random.uniform()
  if result < epsilon:
    return env.action_space.sample()
  else:
    qs = main_nn(state).cpu().data.numpy()
    return np.argmax(qs)

#%%
class UniformBuffer(object):
  def __init__(self, size, device):
    self._size = size
    self.buffer = []
    self.device = device
    self._next_idx = 0

  def add(self, state, action, reward, next_state, done):
    if self._next_idx >= len(self.buffer):
      self.buffer.append((state, action, reward, next_state, done))
    else:
      self.buffer[self._next_idx] = (state, action, reward, next_state, done)
    self._next_idx = (self._next_idx + 1) % self._size

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = torch.as_tensor(np.array(states), device=self.device)
    actions = torch.as_tensor(np.array(actions), device=self.device)
    rewards = torch.as_tensor(np.array(rewards, dtype=np.float32),
                              device=self.device)
    next_states = torch.as_tensor(np.array(next_states), device=self.device)
    dones = torch.as_tensor(np.array(dones, dtype=np.float32),
                            device=self.device)
    return states, actions, rewards, next_states, dones

def train_step(states, actions, rewards, next_states, dones):
  next_qs_argmax = main_nn(next_states).argmax(dim=-1, keepdim=True)
  masked_next_qs = target_nn(next_states).gather(1, next_qs_argmax.type(torch.int64)).squeeze()
  target = rewards + (1.0 - dones) * discount * masked_next_qs
  masked_qs = main_nn(states).gather(1, actions.type(torch.int64).unsqueeze(dim=-1)).squeeze()
  loss = loss_fn(masked_qs, target.detach())
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

def to_gray(observation):
  return np.array(np.dot(observation, [0.299, 0.587, 0.114]), dtype=np.float32)

#%%
# Hyperparameters.
num_episodes = 1200 # @param {type:"integer"}
epsilon = 1.00 # @param {type:"number"}
batch_size =  16# @param {type:"integer"}
discount = 0.99 # @param {type:"number"}
buffer_size = 50000 # @param {type:"integer"}

# 4475 + 900 + 900 +450random2 +450env +450random2 + 450random + 1200taketurns

#%%
buffer = UniformBuffer(size=buffer_size, device=device)

#%%
my_frameQue = deque([], maxlen=4)
last_100_ep_rewards, cur_frame = [], 0
for episode in range(num_episodes+1):
  obs1 = env.reset()
  obs1 = to_gray(obs1)
  for i in range(4):
    my_frameQue.append(obs1)
  ep_reward, done = 0, False
  while not done:
    state = list(my_frameQue)
    state = np.array(state)
    state_in = torch.FloatTensor(np.expand_dims(state / 255., axis=0)).to(device)
    action = select_epsilon_greedy_action(state_in, epsilon)

    next_state, reward, done, info = env.step(action)
    ep_reward += reward
    reward = np.sign(reward)

    obs1 = to_gray(next_state)

    my_frameQue.append(obs1)
    buffer.add(state, action, reward, np.array(list(my_frameQue)), done)
    
    cur_frame += 1
    if epsilon > 0.01:
      epsilon -= 1.1e-6

    if len(buffer) >= batch_size:
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      states = states.type(torch.FloatTensor).to(device) / 255.
      next_states = next_states.type(torch.FloatTensor).to(device) / 255.
      loss = train_step(states, actions, rewards, next_states, dones)

    if cur_frame % 10000 == 0:
      target_nn.load_state_dict(main_nn.state_dict())

  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)

  if episode % 25 == 0:
    print(f'Episode: {episode}/{num_episodes}, Epsilon: {epsilon:.3f}, '\
          f'Loss: {loss:.8f}, Return: {np.mean(last_100_ep_rewards):.4f}')
    if episode > 0:
      torch.save(target_nn, "dqn_weights.pt")
      torch.save(optimizer.state_dict(), "optimizer_weights.pt")
env.close()

#%%
training_bot = DuelingDQN(num_actions).to(device)
training_bot.load_state_dict(torch.load("dqn_weights.pt").state_dict())

my_frameQue = deque([], maxlen=4)
bot_frameQue = deque([], maxlen=4)
last_100_ep_rewards, cur_frame = [], 0
for episode in range(num_episodes+1):
  obs1 = env.reset()
  obs1 = to_gray(obs1)
  obs2 = obs1
  for i in range(4):
    my_frameQue.append(obs1)
    bot_frameQue.append(obs2)
  ep_reward, done = 0, False
  while not done:
    state = list(my_frameQue)
    state = np.array(state)
    state_in = torch.FloatTensor(np.expand_dims(state / 255., axis=0)).to(device)
    action = select_epsilon_greedy_action(state_in, epsilon)
    bot_state = list(bot_frameQue)
    bot_state = np.array(bot_state)
    bot_state = torch.FloatTensor(np.expand_dims(bot_state / 255., axis=0)).to(device)
    bot_action = np.argmax(target_nn(bot_state).cpu().data.numpy())

    next_state, reward, done, info = env.step(action, bot_action)
    ep_reward += reward
    reward = np.sign(reward)

    obs1 = to_gray(next_state)
    obs2 = to_gray(info['otherObs'])
    my_frameQue.append(obs1)
    bot_frameQue.append(obs2)
    buffer.add(state, action, reward, np.array(list(my_frameQue)), done)

    if reward != 0:
      my_frameQue = deque([], maxlen=4)
      bot_frameQue = deque([], maxlen=4)
      for i in range(4):
        my_frameQue.append(obs1)
        bot_frameQue.append(obs2)

    cur_frame += 1
    if epsilon > 0.01:
      epsilon -= 1.1e-6

    if len(buffer) >= batch_size:
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      states = states.type(torch.FloatTensor).to(device) / 255.
      next_states = next_states.type(torch.FloatTensor).to(device) / 255.
      loss = train_step(states, actions, rewards, next_states, dones)

    if cur_frame % 10000 == 0:
      target_nn.load_state_dict(main_nn.state_dict())

  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)

  if episode % 25 == 0:
    print(f'Episode: {episode}/{num_episodes}, Epsilon: {epsilon:.3f}, '\
          f'Loss: {loss:.8f}, Return: {np.mean(last_100_ep_rewards):.2f}')
    if episode > 0:
        torch.save(target_nn, "dqn_weights.pt")
        torch.save(optimizer.state_dict(), "optimizer_weights.pt")
env.close()