'''Solves Pong with Policy Gradients in Tensorflow.'''
# written October 2016 by Sam Greydanus
# inspired by karpathy's gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
import numpy as np
import gym
import tensorflow as tf
import timeit

# hyperparameters
n_obs = 80 * 80           # dimensionality of observations
h = 200                   # number of hidden layer neurons
n_actions = 3             # number of available actions
learning_rate = 1e-4
gamma = .99               # discount factor for reward
decay = 0.99              # decay rate for RMSProp gradients
save_path = 'models/pong.ckpt'

# gamespace
env = gym.make("Pong-v0")  # environment info
obs = env.reset()
prev_x = None
xs, rs, ys = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
batch_size = 10
render = False

# initialize model
model = {}
with tf.variable_scope('layer_one', reuse=False):
  xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
  model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
with tf.variable_scope('layer_two', reuse=False):
  xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
  model['W2'] = tf.get_variable("W2", [h, n_actions], initializer=xavier_l2)

# tf operations
def discount_rewards(r):  # r ~ [game_steps,1]
  discount_f = lambda a, v: a*gamma + v
  r_reverse = tf.scan(discount_f, tf.reverse(r, [True, False]))
  discounted_r = tf.reverse(r_reverse, [True, False])
  return discounted_r

def policy_forward(x):  # x ~ [1,D]
  h = tf.matmul(x, model['W1'])
  h = tf.nn.relu(h)
  logp = tf.matmul(h, model['W2'])
  p = tf.nn.softmax(logp)
  return p

# downsampling
def preprocess(I):
  """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195]  # crop
  I = I[::2, ::2, 0]  # downsample by factor of 2
  I[I == 144] = 0  # erase background (background type 1)
  I[I == 109] = 0  # erase background (background type 2)
  I[I != 0] = 1    # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

# tf placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, n_obs], name="X")
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions], name="Y")
EPR = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="EPR")

# tf reward processing (need discounted_epr for policy gradient wizardry)
discounted_epr = discount_rewards(EPR)
mean, variance = tf.nn.moments(discounted_epr, [0], shift=None, name="reward_moments")
discounted_epr -= mean
discounted_epr /= tf.sqrt(variance + 1e-4)

# tf optimizer op
aprob = policy_forward(X)
loss = tf.nn.l2_loss(Y - aprob)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=discounted_epr)
train_op = optimizer.apply_gradients(grads)

# tf graph initialization
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

# try load saved model
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
load_was_success = True  # yes, I'm being optimistic
try:
  save_dir = '/'.join(save_path.split('/')[:-1])
  ckpt = tf.train.get_checkpoint_state(save_dir)
  load_path = ckpt.model_checkpoint_path
  saver.restore(sess, load_path)
except:
  print("no saved model to load. starting new session")
  load_was_success = False
else:
  print("loaded model: {}".format(load_path))
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
  episode_number = int(load_path.split('-')[-1])

start = timeit.default_timer()
# training loop
while True:
  if render:
    env.render()

  # preprocesscess the observations, set input to network to be difference image
  cur_x = preprocess(obs)
  x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
  prev_x = cur_x

  # stochastically sample a policy from the network
  feed = {X: np.reshape(x, (1, -1))}
  prob = sess.run(aprob, feed)
  prob = prob[0, :]
  action = np.random.choice(n_actions, p=prob)
  label = np.zeros_like(prob)
  label[action] = 1

  # step the environment and get new measurements
  obs, reward, done, info = env.step(action + 1)
  reward_sum += reward

  # record game history
  xs.append(x)
  ys.append(label)
  rs.append(reward)

  if done:
    # update running reward
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

    # parameter update
    feed = {X: np.vstack(xs), EPR: np.vstack(rs), Y: np.vstack(ys)}
    _ = sess.run(train_op, feed)

    # print progress console
    print('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))

    # bookkeeping
    xs, rs, ys = [], [], []  # reset game history
    episode_number += 1  # the Next Episode
    obs = env.reset()  # reset env
    reward_sum = 0
    if episode_number % batch_size == 0:
      saver.save(sess, save_path, global_step=episode_number)
      print("SAVED MODEL #{}".format(episode_number))
      stop = timeit.default_timer()
      print("{} seconds per batch of {} episodes".format(stop - start, batch_size))
      start = timeit.default_timer()
