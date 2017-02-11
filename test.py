import tensorflow as tf
import gym

a=tf.constant(1)
b=tf.constant(2)
sess=tf.Session()
print(sess.run(a+b))
env = gym.make('Pong-v0')
env.reset()
env.render()
raw_input("Press Enter to exit...")
