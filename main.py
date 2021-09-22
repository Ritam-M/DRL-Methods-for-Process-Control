import gym
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve

class Process(gym.Env):
    def __init__(self):
      self.Kp = 3.0
      self.taup = 5.0
      self.delta_t = 0.1
      self.steps = 0
      self.count = 0
      self.eps = 1e-5
      setpoint = [18]*300
      self.setpoint = np.array(setpoint)
      self.state = np.array([0,0,0], dtype=np.float32)

      state_high = np.array([10.0,10.0,10.0], dtype=np.float)
      state_low = np.array([0.0,0.0,0.0], dtype=np.float)
      action_high = np.array([100.0], dtype=np.float)
      action_low = np.array([0.0], dtype=np.float)

      self.observation_space = spaces.Box(state_low, state_high, dtype=np.float32)
      self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)


    def process(self,y,t,action):
      # Kp = process gain
      # taup = process time constant
        dydt = -y/self.taup + self.Kp*action/self.taup
        return dydt

    def step(self, action):
      st_2, st_1, st_0 = self.state
      st__1 = odeint(self.process,st_0,[0,self.delta_t],args=(action,))[-1][0]
      self.steps += 1
      reward = -(st__1-self.setpoint[self.steps-1])**2
      # print(reward)
      if abs(reward)<self.eps:
        self.count+=1

      self.state = (st_1, st_0, st__1)
      done = bool(self.steps>=300 or self.count>=15)

      return (np.array(self.state), reward, done, {})


    def reset(self):
      self.state = [0,0,0]
      self.steps = 0
      self.count = 0
      return np.array(self.state)

if __name__ == '__main__':
	setpoint = [18]*300
	env = Process()
	n_games = 200
	fc1_dims = 400
	fc2_dims = 300
	agent = Agent(alpha=0.001, beta=0.001, input_dims = env.observation_space.shape, tau = 0.01, batch_size = 128, n_actions = env.action_space.shape[0], fc1_dims = 400 , fc2_dims = 300)
	f_name = 'Process' + str(agent.alpha) + 'beta_' + str(agent.beta) + '_' + str(n_games) + '_' + 'games' + 'fc1_dims_' + str(fc1_dims) + '_' + 'fc2_dims_' + str(fc2_dims) 
	figure_file = 'plots/' + f_name + '.png'

	best_score = env.reward_range[0]	
	scores = []
	Actions = []
	Observations = []

	for i in tqdm(range(n_games)):
    		obs = env.reset()
    		score = 0
    		done = False
    		agent.noise.reset()
    		Action = []
    		Observation = []
    		while not done:
      			action = agent.choose_action(obs)
      			Action.append(action)
      			Observation.append(obs[-1])
      			obs_, reward, done, info = env.step(action)
      			agent.remember(obs, action, reward, obs_, done)
      			score += reward
      			agent.learn()
      			obs = obs_
    
    	scores.append(score)
    	avg_score = np.mean(scores[-10:])
    
    	if avg_score>best_score:
        	best_score = avg_score
        # agent.save_models()
    	Actions.append(Action)
    	Observations.append(Observation)    

    	print('episode:',i,"score %.2f", score, 'average score %.2f', avg_score)

x = [i+1 for i in range(len(scores))]
# plot_learning_curve(scores, x, figure_file)
