import gym
import highway_env
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
env = gym.make("parking-v0")

class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, learning_rate):
        super(Actor, self).__init__()
        self.input = input_dims
        self.fc1 = nn.Linear(2 * input_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

    def forward(self, data):
        fc_layer1 = F.relu(self.fc1(data))
        fc_layer2 = F.relu(self.fc2(fc_layer1))
        actions = F.tanh(self.fc3(fc_layer2))
        return actions

class Critic(nn.Module):
    def __init__(self, input_dims, n_actions, learning_rate):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(2 * input_dims + n_actions, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

    def forward(self, data1, data2):
        fc_layer1 = F.relu(self.fc1(torch.cat((data1, data2), 1)))
        fc_layer2 = F.relu(self.fc2(fc_layer1))
        value = self.fc3(fc_layer2)
        return value

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class HindsightExperienceReplayMemory(object):
    def __init__(self, memory_size, input_dims, n_actions):
        super(HindsightExperienceReplayMemory, self).__init__()
        self.max_mem_size = memory_size
        self.counter = 0
        self.state_memory = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.action_memory = np.zeros((memory_size, n_actions), dtype=np.float32)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)
        self.goal_memory = np.zeros((memory_size, input_dims), dtype=np.float32)

    def add_experience(self, state, action, reward, next_state, done, goal):
        curr_index = self.counter % self.max_mem_size
        self.state_memory[curr_index] = state
        self.action_memory[curr_index] = action
        self.reward_memory[curr_index] = reward
        self.next_state_memory[curr_index] = next_state
        self.terminal_memory[curr_index] = done
        self.goal_memory[curr_index] = goal
        self.counter += 1

    def get_random_experience(self, batch_size):
        rand_index = np.random.choice(min(self.counter, self.max_mem_size), batch_size, replace=False)
        rand_state = self.state_memory[rand_index]
        rand_action = self.action_memory[rand_index]
        rand_reward = self.reward_memory[rand_index]
        rand_next_state = self.next_state_memory[rand_index]
        rand_done = self.terminal_memory[rand_index]
        rand_goal = self.goal_memory[rand_index]

        return rand_state, rand_action, rand_reward, rand_next_state, rand_done, rand_goal

class DDPGAgent:
    def __init__(self, actor_learning_rate, critic_learning_rate, n_actions,
                 input_dims, gamma, memory_size, batch_size, tau=0.001):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau

        self.actor = Actor(input_dims=input_dims, n_actions=n_actions,
                           learning_rate=actor_learning_rate)

        self.critic = Critic(input_dims=input_dims, n_actions=n_actions,
                             learning_rate=critic_learning_rate)

        self.target_actor = Actor(input_dims=input_dims, n_actions=n_actions,
                                  learning_rate=actor_learning_rate)

        self.target_critic = Critic(input_dims=input_dims, n_actions=n_actions,
                                    learning_rate=critic_learning_rate)

        self.memory = HindsightExperienceReplayMemory(memory_size=memory_size,
                                                          input_dims=input_dims, n_actions=n_actions)

        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions))

    def store_experience(self, state, action, reward, next_state, done, goal):
        self.memory.add_experience(state=state, action=action,
                                   reward=reward, next_state=next_state,
                                   done=done, goal=goal)

    def get_sample_experience(self):
        state, action, reward, next_state, done, goal = self.memory.get_random_experience(
            self.batch_size)

        t_state = torch.tensor(state).to(self.actor.device)
        t_action = torch.tensor(action).to(self.actor.device)
        t_reward = torch.tensor(reward).to(self.actor.device)
        t_next_state = torch.tensor(next_state).to(self.actor.device)
        t_done = torch.tensor(done).to(self.actor.device)
        t_goal = torch.tensor(goal).to(self.actor.device)

        return t_state, t_action, t_reward, t_next_state, t_done, t_goal

    def choose_action(self, observation, goal, train):
        if train:
            if np.random.random() > 0.1:
                state = torch.tensor([np.concatenate([observation, goal])], dtype=torch.float).to(self.actor.device)
                mu = self.actor.forward(state).to(self.actor.device)
                action = mu + torch.tensor(self.ou_noise(), dtype=torch.float).to(self.actor.device)

                self.actor.train()
                selected_action = action.cpu().detach().numpy()[0]
            else:
                selected_action = np.random.uniform(-1, 1, 2)
        else:
            state = torch.tensor([np.concatenate([observation, goal])], dtype=torch.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            action = mu
            selected_action = action.cpu().detach().numpy()[0]

        return selected_action

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        state, action, reward, next_state, done, goal = self.get_sample_experience()
        concat_state_goal = torch.cat((state, goal), 1)
        concat_next_state_goal = torch.cat((next_state, goal), 1)

        target_actions = self.target_actor.forward(concat_state_goal)
        critic_next_value = self.target_critic.forward(concat_next_state_goal, target_actions).view(-1)

        actor_value = self.actor.forward(concat_state_goal)
        critic_value = self.critic.forward(concat_state_goal, action)

        critic_value[done] = 0.0

        target = (reward + self.gamma * critic_next_value).view(self.batch_size, -1)

        loss_critic = self.critic.loss(target, critic_value)
        loss_critic.backward()
        self.critic.optimizer.step()

        loss_actor = -torch.mean(self.critic.forward(concat_state_goal, actor_value))
        loss_actor.backward()
        self.actor.optimizer.step()

        actor_parameters = dict(self.actor.named_parameters())
        critic_parameters = dict(self.critic.named_parameters())
        target_actor_parameters = dict(self.target_actor.named_parameters())
        target_critic_parameters = dict(self.target_critic.named_parameters())

        for i in actor_parameters:
            actor_parameters[i] = self.tau * actor_parameters[i].clone() + (1 - self.tau) * target_actor_parameters[i].clone()

        for i in critic_parameters:
            critic_parameters[i] = self.tau * critic_parameters[i].clone() + (1 - self.tau) * target_critic_parameters[i].clone()

        self.target_actor.load_state_dict(actor_parameters)
        self.target_critic.load_state_dict(critic_parameters)

    def save_model(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            }, './107081028_hw3_parking_data.pt')

    def load_model(self):
        ckpt = torch.load('./107081028_hw3_parking_data.pt')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_actor.load_state_dict(ckpt['target_actor'])
        self.target_critic.load_state_dict(ckpt['target_critic'])

size = 5
n_episodes = 10000
print(n_episodes)
episodes = []
win_percent = []
success = 0
load_checkpoint = False

agent = DDPGAgent(actor_learning_rate=0.0001, critic_learning_rate=0.0005, n_actions=2,
                            input_dims=6, gamma=0.95,
                            memory_size=10000, batch_size=128)

if load_checkpoint:
    agent.load_model()
load_checkpoint = False

for episode in range(n_episodes):
    obs = env.reset()
    state = obs['observation']
    goal = obs['desired_goal']
    done = False
    transitions = []

    for p in range(50):
        if not done:
            action = agent.choose_action(state, goal, True)
            obs, reward, done, info = env.step(action)
            next_state = obs['observation']
            agent.store_experience(state, action, reward, next_state, done, goal)
            transitions.append((state, action, reward, next_state, info))
            state = next_state
            if done:
                success += 1

    if not done:
        rdm = random.sample(range(0, 50), 20)
        rdm.sort()
        # print(rdm)
        for q in range(20):
            if q != 19:
                p = rdm[random.randint(q, 19)]
                new_goal = transitions[p][0]
            else:
                new_goal = np.copy(state)
            transition = transitions[rdm[q]]
            new_reward = env.compute_reward(transition[3], new_goal, transition[4])
            agent.store_experience(transition[0], transition[1], new_reward,
                                transition[3], np.array_equal(transition[3], new_goal), new_goal)
    for i in range(50):
        agent.learn()

    if episode > 0 and episode % 100 == 0:
        print('success rate for last 100 episodes after', episode, ':', success)
        agent.save_model()
        episodes.append(episode)
        win_percent.append(success / 100)
        success = 0

print('Episodes:', episodes)
print('Win percentage:', win_percent)