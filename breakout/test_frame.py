import gym
import pickle
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import cv2

FRAMESKIP = 4
FRAMES = 4


class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class NoopResetEnv(gym.Wrapper):
    '''From: https://github.com/BITminicc/OpenAI-gym-Breakout/blob/master/atari_wrappers.py'''

    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        #assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


def pre_process_env(env):
    env = NoopResetEnv(env)
    env = SkipFrame(env, skip=FRAMESKIP)
    env = gym.wrappers.gray_scale_observation.GrayScaleObservation(env)
    env = gym.wrappers.resize_observation.ResizeObservation(env, 84)
    env = gym.wrappers.frame_stack.FrameStack(env, FRAMES)
    return env


def act_randomly(env):
    states = []
    done = False
    state = env.reset()
    states.append(state)

    i = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)  # step the environment
        if info['lives'] < 5:
            done = True
            idx = i
        print(i, reward, done, info)
        states.append(state)
        i += 1
    return states, idx


# MAIN
ENV = 'ALE/Breakout-v5'
env = gym.make(ENV, full_action_space=False, frameskip=1, repeat_action_probability=0.01)
env = pre_process_env(env)

states, idx = act_randomly(env)
print(len(states))
print(idx)
#idx = 20
state = states[idx]

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5), constrained_layout=True)
axs[0, 0].imshow(state[0], cmap='gray')
axs[0, 1].imshow(state[1], cmap='gray')
axs[1, 0].imshow(state[2], cmap='gray')
axs[1, 1].imshow(state[3], cmap='gray')
fig.suptitle('4 frames')
plt.show()

#plt.imshow(state_gray)
#plt.imshow(crop(state))
#plt.show()
#print(state[0].squeeze().shape)
#plt.imshow(state[0])
#plt.imshow(state[1].squeeze())
#plt.imshow(state[2].squeeze())
#plt.imshow(state[3].squeeze())
#plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()
