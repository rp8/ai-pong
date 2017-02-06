""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
from gym import wrappers

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
model = pickle.load(open('model.p', 'rb'))
recording = False

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p

env = gym.make("Pong-v0")
if recording:
  env = wrappers.Monitor(env, './data', force=True)

observation = env.reset()
prev_x = None # used in computing the difference frame
reward_sum = 0

while True:
  env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  prob = policy_forward(x)
  action = 2 if np.random.uniform() < prob else 3 # roll the dice!

  observation, reward, done, info = env.step(action)
  reward_sum += reward

  if done:
    print('game finished: reward = %f' % reward + ('' if reward == 1 else '!'))
    env.reset()
