import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output
from IPython.display import display

import gym
import torch
from soft_actor_critic import PolicyNetwork
from pendulumn_sac import NormalizedActions

#Env stuff
env = NormalizedActions(gym.make('Pendulum-v0'))

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]
hidden_dim = 256

policy_net = PolicyNetwork(state_dim,action_dim,hidden_dim)
checkpoint = torch.load('checkpoint.pth')
policy_net.load_state_dict(checkpoint['policy_state_dict'])

#visulazing the trained algo
def display_frames_as_gif(frames):

	# fig plt.subplot()
	patch = plt.imshow(frames[0])
	plt.axis('off')

	def animate(i):
		print(f'Setting Frame {i+1	} of {len(frames)}')
		patch.set_data(frames[i])

	anim = animation.FuncAnimation(plt.gcf(),animate,frames=len(frames),interval=50)
	anim.save('animation.gif', writer='PillowWriter', fps=30)
	display(anim)

state = env.reset()
cum_reward = 0
frames = []

if __name__ == '__main__':
	#Running Test
	for i in range(10000):
		#Render to buffer
		frames.append(env.render(mode='rgb_array'))
		with torch.no_grad():
			action = policy_net.get_action(state).detach()
			state,reward,done,info = env.step(action.numpy())

		if done:
			break

	env.close()
	display_frames_as_gif(frames)