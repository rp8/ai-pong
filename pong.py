""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import cPickle as pickle
import timeit
from pathlib import Path
import gym
import numpy as np

# hyperparameters
H = 200  # number of hidden layer neurons
BATCH_SIZE = 10  # every how many episodes to do a param update?
LEARNING_RATE = 1e-4
GAMMA = 0.99  # discount factor for reward
DECAY_RATE = 0.99  # decay factor for RMSProp leaky sum of grad^2
RESUME = True  # RESUME from previous checkpoint?
RENDER = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
MODEL_FILE = Path('model.p')

def sigmoid(val):
  """ sigmod """
  return 1.0 / (1.0 + np.exp(-val))  # sigmoid "squashing" function to interval [0,1]

def prepro(images):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  images = images[35:195]  # crop
  images = images[::2, ::2, 0]  # downsample by factor of 2
  images[images == 144] = 0  # erase background (background type 1)
  images[images == 109] = 0  # erase background (background type 2)
  images[images != 0] = 1  # everything else (paddles, ball) just set to 1
  return images.astype(np.float).ravel()

def discount_rewards(reward):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(reward)
  running_add = 0
  for val in reversed(xrange(0, reward.size)):
    if reward[val] != 0:
      running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * GAMMA + reward[val]
    discounted_r[val] = running_add
  return discounted_r

def policy_forward(model, inputs):
  """ forward pass """
  hidden = np.dot(model['W1'], inputs)
  hidden[hidden < 0] = 0  # ReLU nonlinearity
  logp = np.dot(model['W2'], hidden)
  probability = sigmoid(logp)
  return probability, hidden  # return probability of taking action 2, and hidden state

def policy_backward(model, epx, eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dw2 = np.dot(eph.T, epdlogp).ravel()
  deltah = np.outer(epdlogp, model['W2'])
  deltah[eph <= 0] = 0  # backpro prelu
  dw1 = np.dot(deltah.T, epx)
  return {'W1': dw1, 'W2': dw2}

def train():
  """ train the network """
  if RESUME and MODEL_FILE.is_file():
    model = pickle.load(open('model.p', 'rb'))
  else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

  # update buffers that add up gradients over a batch
  grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
  rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory

  env = gym.make("Pong-v0")
  observation = env.reset()
  prev_x = None  # used in computing the difference frame
  x_s, h_s, dlogp_s, dr_s = [], [], [], []
  running_reward = None
  reward_sum = 0
  episode_number = 0
  start = timeit.default_timer()

  while True:
    if RENDER:
      env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    deltax = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, hidden_s = policy_forward(model, deltax)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    x_s.append(deltax)  # observation
    h_s.append(hidden_s)  # hidden state
    outputs = 1 if action == 2 else 0  # a "fake label"
    # grad that encourages the action that was taken to be taken (see
    # http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogp_s.append(outputs - aprob)

    # step the environment and get new measurements
    observation, reward, done = env.step(action)
    reward_sum += reward

    # record reward (has to be done after we call step() to get reward for previous action)
    dr_s.append(reward)

    if done:  # an episode finished
      # stack together all inputs, hidden states, action gradients, and rewards for this episode
      ep_x = np.vstack(x_s)
      ep_h = np.vstack(h_s)
      epdlog_p = np.vstack(dlogp_s)
      ep_r = np.vstack(dr_s)
      x_s, h_s, dlogp_s, dr_s = [], [], [], []  # reset array memory

      # compute the discounted reward backwards through time
      discounted_epr = discount_rewards(ep_r)
      # standardize the rewards to be unit normal (helps control the gradient estimator variance)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)

      epdlog_p *= discounted_epr
      # modulate the gradient with advantage (PG magic happens right here.)
      grad = policy_backward(model, ep_x, ep_h, epdlog_p)
      for k in model:
        grad_buffer[k] += grad[k]  # accumulate grad over batch

      # perform rmsprop parameter update every BATCH_SIZE episodes
      if episode_number % BATCH_SIZE == 0:
        for key, val in model.iteritems():
          grad = grad_buffer[key]  # gradient
          rmsprop_cache[key] = DECAY_RATE * rmsprop_cache[key] + (1 - DECAY_RATE) * grad**2
          model[key] += LEARNING_RATE * grad / (np.sqrt(rmsprop_cache[key]) + 1e-5)
          grad_buffer[key] = np.zeros_like(val)  # reset batch gradient buffer

      # boring book-keeping
      episode_number += 1
      if running_reward is None:
        running_reward = reward_sum
      else:
        running_reward = running_reward * 0.99 + reward_sum * 0.01

      print('episode %i reward total = %f, running mean = %f' %
            (episode_number, reward_sum, running_reward))
      if episode_number % 100 == 0:
        pickle.dump(model, open('model.p', 'wb'))
        stop = timeit.default_timer()
        print("average %f seconds per batch of %d episodes" % (stop - start, BATCH_SIZE))
        start = timeit.default_timer()
      reward_sum = 0
      observation = env.reset()  # reset env
      prev_x = None

train()
