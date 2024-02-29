from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import random
#from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F

print("début import")
env = TimeLimit(
	env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


print("import fini")

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.



class ReplayBuffer:
	def __init__(self, capacity, device):
		self.capacity = int(capacity) # capacity of the buffer
		self.data = []
		self.index = 0 # index of the next cell to be filled
		self.device = device
	def append(self, s, a, r, s_, d):
		if len(self.data) < self.capacity:
			self.data.append(None)
		self.data[self.index] = (s, a, r, s_, d)
		self.index = (self.index + 1) % self.capacity
	def sample(self, batch_size):
		batch = random.sample(self.data, batch_size)
		return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
	def __len__(self):
		return len(self.data)

class QNetwork(nn.Module):
	def __init__(self, env):
		super().__init__()
		print(env.observation_space.shape)
		print(env.action_space.shape)
		state_dim = env.observation_space.shape[0]
		print(env.action_space)
		action_dim = env.action_space.n
		self.fc1 = nn.Linear(state_dim + action_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 1)
	def forward(self, x, a):
		x = torch.cat([x, a], 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x




class policyNetwork(nn.Module):
	def __init__(self, env):
		super().__init__()
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.n
		self.fc1 = nn.Linear(state_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc_mu = nn.Linear(256, action_dim)
		# action rescaling
	#	self.register_buffer(
	#		"action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
	#	)
	#	self.register_buffer(
	#		"action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
	#	)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = torch.tanh(self.fc_mu(x))
		return x * self.action_scale + self.action_bias


# ENJOY!
class ProjectAgent:



	def __init__(self ):
		# networks
		device = "cuda" if next(Qfunction.parameters()).is_cuda else "cpu"
		self.scalar_dtype = next(Qfunction.parameters()).dtype
		self.Qfunction = Qfunction
		self.Q_target = deepcopy(self.Qfunction).to(device)
		self.pi = policy_network
		self.pi_target = deepcopy(self.pi).to(device)
		# parameters
		self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
		buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
		self.memory = ReplayBuffer(buffer_size, device)
		self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
		lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
		self.Q_optimizer = torch.optim.Adam(list(self.Qfunction.parameters()), lr=lr)
		self.pi_optimizer = torch.optim.Adam(list(self.pi.parameters()), lr=lr)
		self.tau = config['tau'] if 'tau' in config.keys() else 0.005
		self.exploration_noise = config['exploration_noise'] if 'exploration_noise' in config.keys() else 0.005
		self.delay_learning = config['delay_learning'] if 'delay_learning' in config.keys() else 1e4
		self.tqdm_disable = config['tqdm_disable'] if 'tqdm_disable' in config.keys() else True
		self.disable_episode_report = config['disable_episode_report'] if 'disable_episode_report' in config.keys() else True

	def train(self, env, max_steps):
		x,_ = env.reset()
		episode = 0
		episode_cum_reward = 0
		episode_return = []

		for time_step in range(int(max_steps)):
			# step (policy + noise), add to rb
			if time_step > self.delay_learning:
				with torch.no_grad():
					a = self.pi(torch.tensor(x,dtype=self.scalar_dtype))
					a += torch.normal(0, self.pi.action_scale * self.exploration_noise)
					a = a.cpu().numpy().clip(env.action_space.low, env.action_space.high)
			else:
				a = env.action_space.sample()
			y, r, done, trunc, _ = env.step(a)
			self.memory.append(x,a,r,y,done)
			episode_cum_reward += r
			
			# gradient step
			if time_step > self.delay_learning:
				X, A, R, Y, D = self.memory.sample(self.batch_size)
				## Qfunction update
				with torch.no_grad():
					next_actions = self.pi_target(Y)
					QYA = self.Q_target(Y, next_actions)
					#target = torch.addcmul(R, 1-D, QY, value=self.gamma)
					target = R + self.gamma * (1-D) * QYA.view(-1)
				QXA = self.Qfunction(X, A).view(-1)
				Qloss = F.mse_loss(QXA,target)
				self.Q_optimizer.zero_grad()
				Qloss.backward()
				self.Q_optimizer.step()
				## policy update
				pi_loss = -self.Qfunction(X, self.pi(X)).mean()
				self.pi_optimizer.zero_grad()
				pi_loss.backward()
				self.pi_optimizer.step()
				
				# target networks update
				for param, target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				for param, target_param in zip(self.Qfunction.parameters(), self.Q_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				
			# if done, print episode info
			if done or trunc:
				x, _ = env.reset()
				episode_return.append(episode_cum_reward)
				if not self.disable_episode_report:
					print("Episode ", '{:2d}'.format(episode), 
						  ", buffer size ", '{:4d}'.format(len(self.memory)), 
						  ", episode return ", '{:4.1f}'.format(episode_cum_reward), 
						  sep='')
				episode += 1
				episode_cum_reward = 0
			else:
				x=y
		return episode_return










	def act(self, observation, use_random=False):


		if(use_random):
			return env.action_space.sample()
		
		a = policy(torch.tensor(observation,dtype=scalar_dtype)).numpy()
		res,_,_,_,_ = test_env.step(a)
		return res

	def save(self, path):
		torch.save(self.Qfunction.state_dict(),path + "Qfunction.pth") 
		torch.save(self.pi.state_dict(),path + "pi.pth") 


	def load(self):
		self.Qfunction.load_state_dict(torch.load("save_Qfunction.pth"))
		self.pi.load_state_dict(torch.load("save_pi.pth"))




config = {'gamma': .99,
		  'buffer_size': 1e6,
		  'learning_rate': 3e-4,
		  'batch_size': 256,
		  'tau': 0.005,
		  'delay_learning': 1e4,
		  'exploration_noise': .1,
		  'tqdm_disable': False
		 }
#max_episode_steps = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Qfunction = QNetwork(env).to(device)
policy = policyNetwork(env).to(device)

#agent = ProjectAgent(config, Qfunction, policy)
#episode_returns = agent.train(env, max_episode_steps)

#agent.save("save_")

