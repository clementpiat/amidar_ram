import gym
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import argparse

from agent import Agent
from net import DeepQNet

def train(n_episodes=20, n_steps_max=2000, print_ever_k_episodes=5):
    env = gym.make('Boxing-ram-v0')
    net = DeepQNet()
    agent = Agent(net, env)

    env.reset()
    scores = []
    best_agent, best_score = None, 0

    for i_episode in range(n_episodes):
        cumulative_reward = 0
        observation = env.reset()
        for t in range(n_steps_max):
            action = agent.act(observation, i_episode)
            previous_observation = observation
            observation, reward, done, info = env.step(action)    
            cumulative_reward += reward
            
            agent.remember(previous_observation, action, reward, observation)
        
            if done:
                break
        
        if cumulative_reward > best_score:
            best_score = cumulative_reward
            best_agent = agent

        agent.learn()   
                
        scores.append(cumulative_reward)
        if (i_episode+1)%print_ever_k_episodes==0:
            print(f"Mean score over {print_ever_k_episodes} last episodes: {int(np.mean(scores[-print_ever_k_episodes:]))}")
            
    env.close()
    plt.plot(scores)
    plt.savefig("scores.png")
    
    return best_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--n_episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("-s", "--n_steps_max", type=int, default=2000, help="maximum number of steps")
    parser.add_argument("-k", "--print_ever_k_episodes", type=int, default=5)
    args = parser.parse_args()

    train(n_episodes=args.n_episodes, n_steps_max=args.n_steps_max, print_ever_k_episodes=args.print_ever_k_episodes)