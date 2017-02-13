""" down sample images """
import cPickle as pickle
import numpy as np
import gym
import tensorflow as tf
import sys
from pathlib import Path

def down_sample():
  """ down sample images """
  file_name = "frame.p"
  if len(sys.argv) == 2:
    file_name = sys.argv[1]

  if Path(file_name).is_file():
    frame = pickle.load(open(file_name, 'rb'))
    print("frame loaded from file " + file_name)
  else:
    env = gym.make("Pong-v0")
    env.reset()
    frame, reward, done, info = env.step(2)
    print("frame saved to " + file_name)
    pickle.dump(frame, open(file_name, 'wb'))
  
  sess = tf.Session()
  data = preprocess(sess, frame)
  print(data.shape)

def preprocess(sess, frame):
  """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  with sess.as_default():
    frame = tf.slice(frame, [35, 0, 0], [160, 160, 1])
    frame = tf.image.resize_nearest_neighbor([frame], [80, 80])
    frame = frame[0, :, :,  0]
    I = sess.run(frame)
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

down_sample()
