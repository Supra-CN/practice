import time

import gym

env = gym.make("BreakoutNoFrameskip-v4",render_mode='human')

print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)


obs = env.reset()

for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render("human")
    time.sleep(0.1)
env.close()