import gym

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)

print(f'env = {env}')
print(f'env.observation_space = {env.observation_space}')
print(f'env.observation_space.shape = {env.observation_space.shape}')
print(f'env.action_space = {env.action_space}')
print(f'env.action_space.shape = {env.action_space.shape}')
print(f'observation = {observation}')
print(f'info = {info}')



def policy(observation):
    # return env.action_space.sample()
    return


for _ in range(1000):
    env.render()
    action = policy(observation)  # User-defined policy function
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset(return_info=True)

env.close()
