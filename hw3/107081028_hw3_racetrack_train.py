import gym
import highway_env
import sys
import os
import glob
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np
env = gym.make("racetrack-v0")
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init = 0.6):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        print(action_std_init)
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        self.actor = nn.Sequential(
                    nn.Conv2d(state_dim, out_channels=8, kernel_size=3, padding=1),
                    nn.BatchNorm2d(8),
                    nn.LeakyReLU(),
                    nn.Conv2d(8, out_channels=16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                    nn.Linear(2304, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                    nn.Tanh()
                    )
        
        # critic
        self.critic = nn.Sequential(
                    nn.Conv2d(state_dim, out_channels=8, kernel_size=3, padding=1),
                    nn.BatchNorm2d(8),
                    nn.LeakyReLU(),
                    nn.Conv2d(8, out_channels=16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                    nn.Linear(2304, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError    

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}])
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

max_ep_len = 300            
max_training_timesteps = int(1e6)   
print_freq = max_ep_len * 5     
save_model_freq = int(2.5e4)      
action_std = 0.7
action_std_decay_rate = 0.02
action_std_decay_freq = int(2.5e4)
min_action_std = 0.1

update_timestep = max_ep_len * 4
K_epochs = 40
eps_clip = 0.2 
gamma = 0.9

lr_actor = 0.0003
lr_critic = 0.0005

random_seed = 0
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
checkpoint_path = '107081028_hw3_racetrack_data.pth'
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
time_step = 0
i_episode = 0
print_running_reward = 0
print_running_episodes = 0
while time_step <= max_training_timesteps:
    state = env.reset()
    current_ep_reward = 0
    for t in range(1, max_ep_len + 1):
        action = ppo_agent.select_action([state])
        state, reward, done, _ = env.step(action)
        
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step +=1
        current_ep_reward += reward

        if time_step % update_timestep == 0:
            ppo_agent.update()

        if time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        if time_step % print_freq == 0:
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            print_running_reward = 0
            print_running_episodes = 0
            
        if time_step % save_model_freq == 0:
            ppo_agent.save(checkpoint_path)
           
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1
    i_episode += 1
env.close()