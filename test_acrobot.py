from acrobot import AcrobotEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

torch.manual_seed(0) # set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, input_l = 6, hidden_l1=64, hidden_l2=32, output_l = 1, bias = True): #,
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_l, hidden_l1)
        self.fc2 = nn.Linear(hidden_l1, hidden_l2)
        self.fc3 = nn.Linear(hidden_l2, output_l)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
       	
        return action.data.numpy()[0], m.log_prob(action)

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.01)#, momentum = 0.9)

def get_discounted_returns(rewards, gamma):
    disc_return = rewards    
    for  num, reward in enumerate(rewards):
        disc_return[num] = sum([r2*math.pow(gamma, num2) for num2,r2 in enumerate(rewards[num:])])
    
    return disc_return


def reinforce(env, num_episodes, gamma):
    sum_rewards = np.zeros(num_episodes)
    
    for i in range(num_episodes):
        
        log_probs = []
        rewards = []
        state = env.reset()
        
        done = False
        episode_rewards = 0

        while not done:
            action, log_prob = policy.act(state)
          
            
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
        
            rewards = rewards + [reward]
            episode_rewards+=reward       
        
        rewards = get_discounted_returns(rewards, gamma)
        print("episode", i, "state",state, "sum of rewards", episode_rewards)
        
        sum_rewards[i] = episode_rewards
        sum_of_log_loss = []
        
        for log_prob, reward in zip(log_probs, rewards):
            sum_of_log_loss.append(-log_prob * reward)
       
        loss = torch.cat(sum_of_log_loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    plt.plot(sum_rewards)
    plt.title("Sum of Rewards in each Episode")
    plt.xlabel("Episode")
    plt.ylabel("Sum of returns")
    plt.show()
    
    return policy

if __name__ == "__main__":
	
	env = AcrobotEnv()
	
	#policy = reinforce(env, num_episodes=300, gamma=0.99)
	
	'''
	filesSaveObject = open("policy_nn_best.obj", "wb")
	pickle.dump(policy, filesSaveObject)
	'''
	fileReadObject = open("policy_nn_best.obj", "rb")
	policy = pickle.load(fileReadObject)
	
	state = env.reset()
	done = False
	num_steps = 0
	
	while not done:
		action, _ = policy.act(state)
		print(action)
		state, reward, done, _  = env.step(action)
		env.render()
		num_steps += 1
	
	print(num_steps)

	env.close()
	
