import time
import matplotlib.pyplot as plt

import numpy as np

import gym

env = gym.make('MountainCar-v0')

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

import matplotlib.pyplot as plt

# reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs))

print(f"Upper Bound for Env Observation type({type(env.observation_space.high)}) = {env.observation_space.high}")
print(f"Lower Bound for Env Observation type({type(env.observation_space.low)}) = {env.observation_space.low}")


def mid(a, b, i):
    return (a[i] + b[i]) / 2.


mid_x = mid(env.observation_space.high, env.observation_space.low, 0)
mid_y = mid(env.observation_space.high, env.observation_space.low, 1)

print(f"mid_x Bound for Env Observation type[{mid_x}] = {mid_x}")
print(f"mid_y Bound for Env Observation type[{mid_y}] = {mid_y}")

# Number of steps you run the agent for
num_steps = 1500

obs = env.reset()


def policy(observation):
    m = mid_y
    if observation[1] < m:
        return 0
    elif observation[1] > m:
        return 2
    return 1

    # return observation[0]
    # env.action_space.sample()


for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs)
    # action = env.action_space.sample()

    action = policy(obs)

    # apply the action
    obs, reward, done, info = env.step(action)

    if done or step % 100 == 0:
        print("step ================> {}".format(step))
        print("The new action      is {}".format(action))
        print("The new observation is {}".format(obs))
        print("The new obs.shape   is {}".format(obs.shape))
        print("The new reward      is {}".format(reward))
        print("The new done        is {}".format(done))
        print("The new info        is {}".format(info))

    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    # time.sleep(0.001)

    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()
