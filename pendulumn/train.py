import math
import random

import gym
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from soft_actor_critic import ReplayBuffer,ValueNetwork,SoftQNetwork,PolicyNetwork

torch.autograd.set_detect_anomaly(True)

#As actions are continuous and unbouned we need to clip them
#learn more about wrapper classes here
#https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/
class NormalizedActions(gym.ActionWrapper):
	def action(self,action):
		low = self.action_space.low
		high = self.action_space.high

		action = low + (action+1.0)*0.5*(high-low)
		action = np.clip(action,low,high)

		return action

	def reverse_action(self,action):
		low = self.action_space.low
		high = self.action_space.high
		
		action = 2*(action-low)/(high-low)-1
		action = np.clip(acton,low,high)

		return action

def plot(frame_idx,rewards):
	clear_output(True)
	plt.figure(figsize=(20,5))
	plt.subplot(1,3,1)#num_rows, num_columns, plot_number
	plt.title(f'frame {frame_idx} reward {rewards[-1]}')
	plt.plot(rewards)
	plt.show()


#env stuff
env = NormalizedActions(gym.make('Pendulum-v0'))

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]
hidden_dim = 256

#value function
value_net = ValueNetwork(state_dim,hidden_dim)
target_value_net = ValueNetwork(state_dim,hidden_dim)

# using two q func to minimise overestimation 
# and choose the minimum of the two nets
soft_q_net1 = SoftQNetwork(state_dim,action_dim,hidden_dim)
soft_q_net2 = SoftQNetwork(state_dim,action_dim,hidden_dim)

#policy function
policy_net = PolicyNetwork(state_dim,action_dim,hidden_dim)

for target_param, param in zip(target_value_net.parameters(),value_net.parameters()):
	target_param.data.copy_(param.data)

#loss
value_criterion = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

lr = 3e-4

#optimizer
value_optimizer = optim.Adam(value_net.parameters(),lr=lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(),lr=lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(),lr=lr)
policy_opimizer = optim.Adam(policy_net.parameters(),lr=lr)

#buffer
replay_buffer_size = 10e6
replay_buffer = ReplayBuffer(replay_buffer_size)

#simulation on aggregated observations
def update(batch_size,gamma=0.99,soft_tau=1e-2):

	state, action, reward, next_state, done = replay_buffer.sample(batch_size)
	state = torch.FloatTensor(state)
	next_state = torch.FloatTensor(next_state)
	action = torch.FloatTensor(action)
	reward = torch.FloatTensor(reward)
	reward = torch.unsqueeze(reward,dim=1)
	done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

	predicted_q_value1 = soft_q_net1(state, action)
	predicted_q_value2 = soft_q_net2(state, action)

	predicted_value = value_net(state)

	new_action, log_prob ,_,_,_ = policy_net.evaluate(state)

	#----Training Q Function----
	target_value = target_value_net(next_state)
	target_q_value = reward+(1-done)*gamma*target_value 

	# loss = (Q(s_t,a_t) - (r + gamma*V(s_t+1)) )**2
	q_value_loss1 = soft_q_criterion1(predicted_q_value1,target_q_value.detach())
	q_value_loss2 = soft_q_criterion2(predicted_q_value2,target_q_value.detach())
	

	soft_q_optimizer1.zero_grad()
	q_value_loss1.backward()
	soft_q_optimizer1.step()

	soft_q_optimizer2.zero_grad()
	q_value_loss2.backward()
	soft_q_optimizer2.step()


	#----Training Value Function----
	predicted_new_q_value = torch.min(soft_q_net1(state,new_action),soft_q_net2(state,new_action))
	target_value_func = predicted_new_q_value - log_prob

	#loss = (V(s_t) - ( Q(s_t,a_t) + H(pi(:,s_t)) ))**2  --H(pi(:,s_t)) = -log(pi(:,s_t))--
	value_loss = value_criterion(predicted_value,target_value_func.detach())

	value_optimizer.zero_grad()
	value_loss.backward()
	value_optimizer.step()


	#----Training Policy Function----
	#maximise (Q(s_t,a_t) + H(pi(:,s_t)) )--> (Q(s_t,a_t) - log(pi(:,s_t)) 
	#minimise (log(pi:,s_t) - Q(s_t,a_t))
	policy_loss = (log_prob - predicted_new_q_value).mean()

	policy_opimizer.zero_grad()
	policy_loss.backward()
	policy_opimizer.step()

	#Update target network parameters
	for target_param, param in zip(target_value_net.parameters(),value_net.parameters()):
		target_param.data.copy_(soft_tau*param + (1-soft_tau)*target_param)


max_frames  = 40000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 32

if __name__ == '__main__':

	while frame_idx < max_frames:
		#number of episodes
		state = env.reset()
		episode_reward = 0
		
		for step in range(max_steps):
		#run until episode end or until max_steps
			if frame_idx > 1000 :
				with torch.no_grad():
					action = policy_net.get_action(state).detach()
					next_state, reward, done, _ = env.step(action.numpy())
			else :
				#random actions
				action = env.action_space.sample()
				next_state, reward, done, _ = env.step(action)

			replay_buffer.push(state,action,reward,next_state,done)
			state = next_state
			episode_reward += reward
			frame_idx += 1
	 
			if len(replay_buffer) > batch_size:
				#run simulation
				update(batch_size)
			
			if frame_idx % 1000 == 0:
				plot(frame_idx, rewards)
			
			if done:
				break

			print(f'steps {frame_idx}/{max_frames} eps_reward {episode_reward}')
			
		rewards.append(episode_reward)

	torch.save({
		'epochs':max_frames,
		'policy_state_dict':policy_net.state_dict(),
		},'checkpoint.pth')