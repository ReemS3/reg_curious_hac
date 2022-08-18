import time

from mujoco.ant_env import AntEnv
import register_envs
from mujoco.ant_reacher.reach import AntReacherEnv


def episode():
    ...


env = AntReacherEnv()
env.render()
time.sleep(0.1)
for t in range(100):
    env.display_end_goal(end_goal=env.goal)
    obs, reward, done, _ = env.step(env.action_space.sample())
    print(reward)
    env.render()
    time.sleep(0.001)
    if done:
        break

env.close()