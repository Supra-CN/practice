'''
https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=zh-cn
'''

from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# Set up a virtual display for rendering OpenAI gym environments.
# 设置一个虚拟显示器，用以渲染 OpenAi gym 环境
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
# display = pyvirtualdisplay.Display(visible=True, size=(1400, 900)).start()

tf.version.VERSION



'''
Hyperparameters
超参数
'''

# num_iterations = 20000 # @param {type:"integer"}
#
num_iterations = 4000 # @param {type:"integer"}
# num_iterations = 2000 # @param {type:"integer"}
# num_iterations = 200 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

gym_env_render_mode = 'human'

'''
Environment
环境
In Reinforcement Learning (RL), an environment represents the task or problem to be solved. 
在强化学习（RL）中，一个环境代表一个需要被解决的任务或问题。
Standard environments can be created in TF-Agents using tf_agents.environments suites. 
可以采用 tf_agents.environments 套件，在 TF-Agents 中创建标准环境。
TF-Agents has suites for loading environments from sources such as the OpenAI Gym, Atari, and DM Control.
TF-Agents 具备从 OpenAI Gym, Atari, 和 DM Control 等来源加载环境的套件

Load the CartPole environment from the OpenAI Gym suite.
从 OpenAi Gym 套件中加载 CartPole 环境
'''

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)

'''
You can render this environment to see how it looks. 
A free-swinging pole is attached to a cart. 
The goal is to move the cart right or left in order to keep the pole pointing up.
'''

env.reset()
# PIL.Image.fromarray(env.render()).show()
env.render(mode=gym_env_render_mode)


'''
The environment.step method takes an action in the environment and returns a TimeStep tuple containing the next observation of the environment and the reward for the action.

The time_step_spec() method returns the specification for the TimeStep tuple. Its observation attribute shows the shape of observations, the data types, and the ranges of allowed values. The reward attribute shows the same details for the reward.
'''

time_step_spec = env.time_step_spec()

print('time step Spec:')
print(time_step_spec)

print('Observation Spec:')
print(time_step_spec.observation)

print('Reward Spec:')
print(time_step_spec.reward)

'''
The action_spec() method returns the shape, data types, and allowed values of valid actions.
'''

print('Action Spec:')
print(env.action_spec())

'''
In the Cartpole environment:

    observation is an array of 4 floats:
        the position and velocity of the cart
        the angular position and velocity of the pole
    reward is a scalar float value
    action is a scalar integer with only two possible values:
        0 — "move left"
        1 — "move right"
'''

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

'''
Usually two environments are instantiated: one for training and one for evaluation.
'''

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

'''
The Cartpole environment, like most environments, is written in pure Python. This is converted to TensorFlow using the TFPyEnvironment wrapper.

The original environment's API uses Numpy arrays. 
The TFPyEnvironment converts these to Tensors to make it compatible with Tensorflow agents and policies.
'''

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
print('train_env')
print(train_env)
print('eval_env')
print(eval_env)

'''
Agent
The algorithm used to solve an RL problem is represented by an Agent. 
TF-Agents provides standard implementations of a variety of Agents, including:

DQN (used in this tutorial)
REINFORCE
DDPG
TD3
PPO
SAC
The DQN agent can be used in any environment which has a discrete action space.

At the heart of a DQN Agent is a QNetwork, 
a neural network model that can learn to predict QValues (expected returns) for all actions, 
given an observation from the environment.

We will use tf_agents.networks. 
to create a QNetwork.
The network will consist of a sequence of tf.keras.layers.Dense layers, 
where the final layer will have 1 output for each possible action.
'''

fc_layer_params = (100, 50) # 定义两层 dense layer 的 units 个数
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
print(f'action_tensor_spec: {action_tensor_spec}')
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
print(f'num_actions: {num_actions}')

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]

print(f'dense_layers: {dense_layers}')
for i, l, in enumerate(dense_layers):
    print(f'dense_layers[{i}]: {l.units}')


q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

print(f'q_net: {q_net}')
# q_net.build(input_shape=time_step_spec.observation.shape)
# print(f'q_net: {q_net.summary()}')

'''
Now use tf_agents.agents.dqn.dqn_agent to instantiate a DqnAgent. 
In addition to the time_step_spec, 
action_spec and the QNetwork, the agent constructor also requires an optimizer (in this case, AdamOptimizer),
a loss function, and an integer step counter.
'''

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


'''
Policies
A policy defines the way an agent acts in an environment. Typically, the goal of reinforcement learning is to train the underlying model until the policy produces the desired outcome.

In this tutorial:

The desired outcome is keeping the pole balanced upright over the cart.
The policy returns an action (left or right) for each time_step observation.
Agents contain two policies:

agent.policy — The main policy that is used for evaluation and deployment.
agent.collect_policy — A second policy that is used for data collection.
'''

eval_policy = agent.policy
collect_policy = agent.collect_policy

print(f'eval_policy: {eval_policy}')
print(f'collect_policy: {collect_policy}')

'''
Policies can be created independently of agents.
For example, 
use tf_agents.policies.random_tf_policy to create a policy which will randomly select an action for each time_step.
'''

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

'''
To get an action from a policy, call the policy.action(time_step) method. 
The time_step contains the observation from the environment. 
This method returns a PolicyStep, which is a named tuple with three components:

action — the action to be taken (in this case, 0 or 1)
state — used for stateful (that is, RNN-based) policies
info — auxiliary data, such as log probabilities of actions
'''

example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load('CartPole-v0'))

time_step = example_environment.reset()

policy_step = random_policy.action(time_step)

print(f'policy_step: {policy_step}')

'''
Metrics and Evaluation
The most common metric used to evaluate a policy is the average return.
The return is the sum of rewards obtained while running a policy in an environment for an episode. Several episodes are run, 
creating an average return.

The following function computes the average return of a policy, given the policy, environment, and a number of episodes.
'''

def compute_avg_return(environment, policy, num_episodes=10, render=False):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
      if(render):
        environment.render(mode=gym_env_render_mode)

    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

'''
Running this computation on the random_policy shows a baseline performance in the environment.
'''

print(f'eval_env with random_policy in episodes({num_eval_episodes}) avg ret: {compute_avg_return(eval_env, random_policy, num_eval_episodes)}')


'''
Replay Buffer
In order to keep track of the data collected from the environment, 
we will use Reverb, an efficient, extensible, and easy-to-use replay system by Deepmind. 
It stores experience data when we collect trajectories and is consumed during training.

This replay buffer is constructed using specs describing the tensors that are to be stored, 
which can be obtained from the agent using agent.collect_data_spec.
'''

table_name = 'uniform_table'

the_agent_collect_data_spec = agent.collect_data_spec

print(f'the_agent_collect_data_spec: {the_agent_collect_data_spec}')
# print(f'the_agent_collect_data_spec._fields: {the_agent_collect_data_spec._fields}')

replay_buffer_signature = tensor_spec.from_spec(
      the_agent_collect_data_spec)

print(f'replay_buffer_signature L1 from_spec: {replay_buffer_signature}')

replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

print(f'replay_buffer_signature L2 add_outer_dim: {replay_buffer_signature}')

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

print(f'reverb_server: {reverb_server}')

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

print(f'replay_buffer: {replay_buffer}')

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

print(f'rb_observer: {rb_observer}')

'''
For most agents, collect_data_spec is a named tuple called Trajectory, containing the specs for observations, actions, rewards, and other items.
'''


'''Data Collection
Now execute the random policy in the environment for a few steps, recording the data in the replay buffer.

Here we are using 'PyDriver' to run the experience collecting loop. You can learn more about TF Agents driver in our drivers tutorial.
'''

print('py driver run')

_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps)

print(f'_driver: {_driver}')

_driver_run_ret = _driver.run(train_py_env.reset())

print(f'_driver_run_ret: {_driver_run_ret}')

'''
The replay buffer is now a collection of Trajectories.
'''
# For the curious:
# Uncomment to peel one of these off and inspect it.
# iter(replay_buffer.as_dataset()).next()


'''
The agent needs access to the replay buffer. This is provided by creating an iterable tf.data.Dataset pipeline which will feed data to the agent.

Each row of the replay buffer only stores a single observation step. But since the DQN Agent needs both the current and next observation to compute the loss, the dataset pipeline will sample two adjacent rows for each item in the batch (num_steps=2).

This dataset is also optimized by running parallel calls and prefetching data.
'''

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

print(f'dataset: {dataset}')

iterator = iter(dataset)
print(f'dataset: {iterator}')


# For the curious:
# Uncomment to see what the dataset iterator is feeding to the agent.
# Compare this representation of replay data
# to the collection of individual trajectories shown earlier.

# iterator.next()
# print(f'iterator.next: {iterator.next()}')


'''
Training the agent
Two things must happen during the training loop:

collect data from the environment
use that data to train the agent's neural network(s)
This example also periodicially evaluates the policy and prints the current score.

The following will take ~5 minutes to run.
'''

# try:
#   %%time
# except:
#   pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
print(f'agent.train L1: {agent.train}')

agent.train = common.function(agent.train)

print(f'agent.train L2 by function: {agent.train}')


print(f'agent.train_step_counter L1: {agent.train_step_counter}')

# Reset the train step.
agent.train_step_counter.assign(0)

print(f'agent.train L2 by assign(0): {agent.train_step_counter}')

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)

print(f'avg_return: {avg_return}')

returns = [avg_return]

print('Reset the environment.')
# Reset the environment.
time_step = train_py_env.reset()

print('Create a driver to collect experience.')
# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

print(f'iterator.size at 0 = {iterator.__sizeof__()}):')
print(f'for _ in range(num_iterations = {num_iterations}):')
for i in range(num_iterations):

  # Collect a few steps and save to the replay buffer.
  # print(f'iterator.size at loop[{i}] start = {iterator.__sizeof__()}):')

  time_step, unused_info_run = collect_driver.run(time_step)

  # print(f'iterator.size at loop[{i}] run = {iterator.__sizeof__()}):')

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)

  # print(f'iterator.size at loop[{i}] nest = {iterator.__sizeof__()}):')

  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print(f'loop = {i}: step = {step}: loss = {train_loss}')
    print(f'    time_step = {time_step}')
    # print(f'    unused_info_run = {unused_info_run}')
    # print(f'    experience = {experience}')
    # print(f'    unused_info = {unused_info}')

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print(f'step = {step}: Average Return = {avg_return}')
    returns.append(avg_return)
    print(f'step = {step}: Average Returns = {returns}')

print(f'for _ in range(num_iterations = {num_iterations}): end')


'''
Visualization
Plots
Use matplotlib.pyplot to chart how the policy improved during training.

One iteration of Cartpole-v0 consists of 200 time steps. The environment gives a reward of +1 for each step the pole stays up, so the maximum return for one episode is 200. The charts shows the return increasing towards that maximum each time it is evaluated during training. (It may be a little unstable and not increase monotonically each time.)
'''

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)

plt.show()

'''
Videos
Charts are nice. But more exciting is seeing an agent actually performing a task in an environment.

First, create a function to embed videos in the notebook.
'''

def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)


'''
Now iterate through a few episodes of the Cartpole game with the agent. The underlying Python environment (the one "inside" the TensorFlow environment wrapper) provides a render() method, which outputs an image of the environment state. These can be collected into a video.
'''

def create_policy_eval_video(policy, filename, num_episodes=10, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      eval_py_env.render(mode=gym_env_render_mode)
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())
        eval_py_env.render(mode=gym_env_render_mode)

  return embed_mp4(filename)

create_policy_eval_video(agent.policy, "trained-agent")

'''
For fun, compare the trained agent (above) to an agent moving randomly. (It does not do as well.)
'''

create_policy_eval_video(random_policy, "random-agent")