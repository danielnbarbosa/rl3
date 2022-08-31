from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import pickle
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import cv2

env = gym_super_mario_bros.make('SuperMarioBros-2-1-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


def save():
    states = []
    done = False
    state = env.reset()
    states.append(state)

    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)  # step the environment
        states.append(state)
        #plt.imshow(state)
        #plt.show()
    #with open('states.pkl', 'wb') as file:
    #    pickle.dump(states, file)
    return states


def load():
    with open('states.pkl', 'rb') as file:
        return pickle.load(file)


def rgb_to_grayscale(state):
    return cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)


def crop(state):
    '''Crop 12 pixels off the bottom and 6 from each side.'''
    return state[32:, :]


def resize(state):
    return cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)


# MAIN
#save()
states = save()
print(len(states))
idx = 0
state = states[idx]
print(state.shape)
state_gray = resize(rgb_to_grayscale(state))
print(state_gray.shape)
#plt.imshow(state_gray)
#plt.imshow(crop(state))
#plt.show()
cv2.imshow('RGB', state)
cv2.imshow('Gray', state_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
