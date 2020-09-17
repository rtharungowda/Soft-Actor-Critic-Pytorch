import math
import random
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal
import gym

#### This is for continous action space only not discrete

class ReplayBuffer(object):
	def __init__(self,capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self,state,action,reward,next_state,done):
		if(len(self.buffer)<self.capacity):
			self.buffer.append(None)

		self.buffer[int(self.position)] = (state,action,reward,next_state,done)
		self.position = (self.position+1)%self.capacity

	def sample(self,batch_size):
		batch = iter(random.sample(self.buffer,batch_size))
		# print(batch[0])
		state,action,reward,next_state,done = map(np.stack,zip(*batch)) #batch[:][0],batch[:][1],batch[:][2],batch_size[:][3],batch[:][4] 
		return state,action,reward,next_state,done

	def __len__(self):
		return len(self.buffer)


class ValueNetwork(nn.Module):
	def __init__(self,state_dim,hidden_dim,init_w=3e-3):
		super(ValueNetwork,self).__init__()
		self.linear1 = nn.Linear(state_dim,hidden_dim)
		self.linear2 = nn.Linear(hidden_dim,hidden_dim)
		self.linear3 = nn.Linear(hidden_dim,1)

		self.linear3.weight.data.uniform_(-init_w,init_w)
		self.linear3.bias.data.uniform_(-init_w,init_w)

	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

class SoftQNetwork(nn.Module):
	def __init__(self,state_dim,action_dim,hidden_size,init_w=3e-3):
		super(SoftQNetwork,self).__init__()
		self.linear1 = nn.Linear(state_dim+action_dim,hidden_size)
		self.linear2 = nn.Linear(hidden_size,hidden_size)
		self.linear3 = nn.Linear(hidden_size,1)

		self.linear3.weight.data.uniform_(-init_w,init_w)
		self.linear3.bias.data.uniform_(-init_w,init_w)

	def forward(self,state,action):
		x = torch.cat((state,action),dim=1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

class PolicyNetwork(nn.Module):
	def __init__(self,state_dim,action_dim,hidden_dim,init_w=3e-3,log_std_min=-20,log_std_max=2):
		super(PolicyNetwork,self).__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.linear1 = nn.Linear(state_dim,hidden_dim)
		self.linear2 = nn.Linear(hidden_dim,hidden_dim)

		self.mean_linear = nn.Linear(hidden_dim,action_dim)
		self.mean_linear.weight.data.uniform_(-init_w,init_w)
		self.mean_linear.bias.data.uniform_(-init_w,init_w)

		self.log_std_linear = nn.Linear(hidden_dim,action_dim)
		self.log_std_linear.weight.data.uniform_(-init_w,init_w)
		self.log_std_linear.bias.data.uniform_(-init_w,init_w)

	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		mean = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)

		return mean,log_std

	def evaluate(self,state,epsilon=1e-6):

		mean,log_std = self.forward(state)
		std = log_std.exp() #exponentiating to get +ve std
		normal = Normal(0,1)
		z = normal.sample()
		action = torch.tanh(mean+std*z)
		#https://pytorch.org/docs/stable/distributions.html
		#log_prob to create a differentiable loss function
		#Normal(mean,sd)l.og_prob(a) finds the normal dist around the passed value 'a'
		log_prob = Normal(mean,std).log_prob(mean+std*z) - torch.log(1-action.pow(2)+epsilon)
		return action, log_prob, z, mean, log_std

	def get_action(self,state):
		state = torch.FloatTensor(state).unsqueeze(0)
		mean, log_std = self.forward(state)
		std = log_std.exp()

		normal = Normal(mean,std)
		z = normal.sample()
		action = torch.tanh(z)
		return action[0]



