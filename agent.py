import torch
from torch import nn
import numpy as np
import copy
import random as rd
from collections import deque

class Agent():
    def __init__(self, net, env, discount=0.9, epsilon=0.99, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, C=1):
        """
        net: network that goes from the state space to the action space
        discount: discount factor in the bellman equation
        epsilon: epsilon factor for the epsilon greedy policy
        C: number of iterations before updating the target network and copying the weights of the current network (for convergence stability)
        """
        self.net = net
        self.target_net = copy.deepcopy(net)
        self.env = env
        
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.C = C
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-3)
        self.loss_fun = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=100000)
        self.iteration = 0
        self.loss = 0
        self.print_loss_every_k_iter = 100

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - np.log10((t + 1) * self.epsilon_decay)))
        
    def act(self, observation, t):
        observation = torch.Tensor(observation)
        q = self.net(observation)
        epsilon = self.epsilon #self.get_epsilon(t)
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()

        return torch.argmax(q).item()
        
    def remember(self, previous_observation, action, reward, observation):
        self.replay_buffer.append((previous_observation, action, reward, observation))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if (self.iteration % self.C) == (self.C - 1):
            """
            Update target network
            """
            self.target_net = copy.deepcopy(self.net)
        
        samples = rd.sample(self.replay_buffer, self.batch_size)
        previous_observations = torch.cat([torch.Tensor(previous_observation).view(1,-1) for previous_observation,_,_,_ in samples])
        observations = torch.cat([torch.Tensor(observation).view(1,-1) for _,_,_,observation in samples])
        rewards = torch.Tensor([reward for _,_,reward,_ in samples])

        q = self.net(previous_observations)
        #q = torch.Tensor([q[i,action] for i, (_,action,_,_) in enumerate(samples)])
        q = q[list(range(len(samples))), [action for (_,action,_,_) in samples]]
        loss = self.loss_fun(q, rewards + self.discount * torch.max(self.target_net(observations), axis=1).values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        self.iteration += 1
        
        self.loss += loss.item()
        if (self.iteration+1)%self.print_loss_every_k_iter==0:
            print(f"Average Loss: {self.loss/self.print_loss_every_k_iter}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
