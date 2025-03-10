# pong
pong.py - copied from [Andrej karpathy](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)  
rl_pong.py - copied from [Sam Greydanus](https://gist.github.com/greydanus/5036f784eec2036252e1990da21eda18)  

## Installation
```bash
$sudo apt-get update
$sudo apt-get install python-pip python-dev python-virtualenv
$virtualenv --system-site-packages ~/tf
$source ~/tf/bin/activate
(tf)$pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
(tf)$pip install keras
(tf)$pip install gym
(tf)$pip install gym[atari]
(tf)$pip install pathlib
(tf)$
```

## Test
test.py
```python
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
```

```bash
(tf)$python test.py
# screen will print a "3" and a pong video game should show up on the screen 
(tf)$deactivate
$
```

## Watch Pong Game
```bash
(tf)$python play.py
```

## Train the Network
```bash
(tf)$python pong.py
```
