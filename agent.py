import torch
from torch import nn
import numpy as np
import copy
import random as rd

class Agent():
    def __init__(self, net, policy="epsilon-greedy", discount=0.999, epsilon=0.3, batch_size=16, C=32):
        """
        net: network that goes from the state space to the action space
        discount: discount factor in the bellman equation
        epsilon: epsilon factor for the epsilon greedy policy
        C: number of iterations before updating the target network and copying the weights of the current network (for convergence stability)
        """
        self.net = net
        self.target_net = copy.deepcopy(net)
        
        self.policy = policy
        self.discount = discount
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.C = C
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()
        self.replay_buffer = []
        self.iteration = 0

        
    def act(self, observation):
        observation = torch.Tensor(observation)
        q = self.net(observation)
        if self.policy == "greedy":
            return torch.argmax(q).item()
        
        elif self.policy == "epsilon-greedy":
            if np.random.random() > self.epsilon:
                return np.random.randint(self.net.action_space_size)
            return torch.argmax(q).item()
        
        else:
            raise BaseException("Not implemented yet.")
        
    def learn(self, previous_observation, action, reward, observation):
        self.replay_buffer.append((previous_observation, action, reward, observation))
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if self.iteration >= self.C:
            """
            Update target network
            """
            self.target_net = copy.deepcopy(self.net)
            self.iteration = 0
        
        samples = rd.sample(self.replay_buffer, self.batch_size)
        previous_observations = torch.cat([torch.Tensor(previous_observation).view(1,-1) for previous_observation,_,_,_ in samples])
        observations = torch.cat([torch.Tensor(observation).view(1,-1) for _,_,_,observation in samples])
        rewards = torch.Tensor([reward for _,_,reward,_ in samples])

        q = self.net(previous_observations)
        q = torch.Tensor([q[i,action] for i, (_,action,_,_) in enumerate(samples)])
        loss = self.loss(q, rewards + self.discount * torch.max(self.target_net(observations), axis=1).values)
            
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iteration += 1
