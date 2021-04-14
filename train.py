import gym
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from agent import Agent
from net import DeepQNet

def train():
    n_episodes = 20
    n_steps_max = 2000
    print_ever_k_episodes = 5

    net = DeepQNet()
    agent = Agent(net)
    
    env = gym.make('Amidar-ram-v0')
    env.reset()
    scores = []

    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(n_steps_max):
            # env.render()
            action = agent.act(observation)
            previous_observation = observation
            observation, reward, done, info = env.step(action)
            agent.learn(previous_observation, action, reward, observation)
            
            if done:
                break
                
        scores.append(t+1)
        if (i_episode+1)%print_ever_k_episodes==0:
            print(f"Mean score over {print_ever_k_episodes} last episodes: {np.mean(scores[-print_ever_k_episodes:])}")
            
    env.close()
    plt.plot(scores)
    plt.savefig("scores.png")

if __name__ == "__main__":
    train()