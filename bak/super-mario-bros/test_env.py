import time
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import numpy as np
import gym
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT


class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            #print('skip', i)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=20)


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
#env = RecordVideo(env, 'video_tmp')
#env = JoypadSpace(env, RIGHT_ONLY)
#env = SkipFrame(env, skip=4)
#env = GrayScaleObservation(env)
#env = ResizeObservation(env, 84)
#env = FrameStack(env, 4)

frames = []
done = False
state = env.reset()
#env.start_video_recorder()

steps = 0
for step in range(1000):
    #env.render()
    #time.sleep(.05)
    frames.append(env.render(mode="rgb_array"))
    #frames.append(state)
    state, reward, done, info = env.step(env.action_space.sample())

    steps += 1
    #print(steps)
    if done:
        break

print('Steps:', steps)
env.close()

save_frames_as_gif(frames)
