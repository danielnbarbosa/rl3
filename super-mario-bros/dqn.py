'''
DQN for learning from pixels.
CNN.
Converts RGB observation to cropped, grayscale stack of frames.
Replay memory.
Fixed Q targets with target network.
Dueling networks.
Double DQN.
Frameskip.
'''

import logging
import random
import time
import pickle
from pathlib import Path
from collections import deque
import argparse
from datetime import datetime
import gym
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import cv2
from torch.utils.tensorboard import SummaryWriter

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
'''
Hyper parameters from DDQN paper:

TRAIN_STEPS_MAX = 50_000_000
REPLAY_MEMORY_MIN = 200_000
REPLAY_MEMORY_SIZE = 1_000_000
SYNC_TARGET_MODEL_EVERY = 10_000
EPS_DECAY_STEPS = 1_000_000
'''

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# CPU Config
if DEVICE == 'cpu':
    PYTORCH_THREADS = None  # might be helpful to set to 8 on M1 so it doesn't use efficency cores

    TRAIN_STEPS_MAX = 5_000_000  # train for this many steps, will go a little beyond to finish the current episode
    REPLAY_MEMORY_MIN = 20_000  # minimum amount of accumulated experience before before we begin sampling
    REPLAY_MEMORY_SIZE = 50_000  # max size of replay memory buffer
    BATCH_SIZE = 32  # number of items to randomly sample from replay memory
    SYNC_TARGET_MODEL_EVERY = 2000  # how often (in steps) to copy weights to target model
    LEARN_EVERY = 4  # update model weights every n steps via gradient descent
    FRAMES = 4  # number of observations to stack together to form the state
    FRAMESKIP = 4  # number of frames to repeat the same actions

    LR = 0.00025  # learning rate
    GAMMA = 0.99  # discount rate
    EPS_START = 1  # starting value of epsilon
    EPS_MIN = .1  # minimum value for epsilon
    EPS_DECAY_STEPS = 200_000  # over how many steps to linearly reduce epsilon until it reaches EPS_MIN

    SHOW_PROGRESS_EVERY = 1  # how often (in episodes) to show training results
    SAVE_MODEL_EVERY = 1000  # how often (in episodes) to save intermediate models
    MOVING_AVERAGE = 100  # moving average window to use when printing intermediate results to console

# GPU Config
elif DEVICE == 'cuda':
    PYTORCH_THREADS = None  # might be helpful to set to 8 on M1 so it doesn't use efficency cores

    TRAIN_STEPS_MAX = 10_000_000  # train for this many steps, will go a little beyond to finish the current episode
    REPLAY_MEMORY_MIN = 200_000  # minimum amount of accumulated experience before before we begin sampling
    REPLAY_MEMORY_SIZE = 1_000_000  # max size of replay memory buffer
    BATCH_SIZE = 32  # number of items to randomly sample from replay memory
    SYNC_TARGET_MODEL_EVERY = 10_000  # how often (in steps) to copy weights to target model
    LEARN_EVERY = 4  # update model weights every n steps via gradient descent
    FRAMES = 4  # number of observations to stack together to form the state
    FRAMESKIP = 4  # number of frames to repeat the same actions

    LR = 0.00025  # learning rate
    GAMMA = 0.99  # discount rate
    EPS_START = 1  # starting value of epsilon
    EPS_MIN = .1  # minimum value for epsilon
    EPS_DECAY_STEPS = 1_000_000  # over how many steps to linearly reduce epsilon until it reaches EPS_MIN

    SHOW_PROGRESS_EVERY = 1  # how often (in episodes) to show training results
    SAVE_MODEL_EVERY = 1000  # how often (in episodes) to save intermediate models
    MOVING_AVERAGE = 100  # moving average window to use when printing intermediate results to console


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
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class Model2Layer(nn.Module):
    """
    Input is 4 stacked 84x84 int8 grayscale frames.
    This is the architecture from Deepmind 2013 paper "Playing Atari with Deep Reinforcement Learning".
    Added dueling networks which doubles the number of params.
    Total params: 1.3M
    """

    def __init__(self, outputs, lr=1e-2):
        super(Model2Layer, self).__init__()
        # yapf: disable
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # (N, 4, 84, 84)  -> (N, 16, 20, 20)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (N, 16, 20, 20) -> (N, 32, 9, 9)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, outputs)
        )
        # yapf: enable
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)

    def forward(self, x):
        assert x.shape == (1, FRAMES, 84, 84) or x.shape == (BATCH_SIZE, FRAMES, 84, 84)
        x = (x / 255.0)  # rescale pixel value as 0 to 1.
        F = self.conv(x)
        V = self.value_stream(F)
        A = self.advantage_stream(F)
        Q = V + (A - A.mean())
        return Q


class Model3Layer(nn.Module):
    """
    Input is 4 stacked 84x84 int8 grayscale frames.
    This is the architecture from Deepmind 2015 paper "Human-level control through deep reinforcement learning"
    Added dueling networks which doubles the number of params.
    Total params: 3.3M
    """

    def __init__(self, outputs, lr=1e-2):
        super(Model3Layer, self).__init__()
        # yapf: disable
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (N, 4, 84, 84)  -> (N, 32, 20, 20)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (N, 32, 20, 20) -> (N, 64, 9, 9)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (N, 64, 9, 9) -> (N, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, outputs)
        )
        # yapf: enable
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)

    def forward(self, x):
        assert x.shape == (1, FRAMES, 84, 84) or x.shape == (BATCH_SIZE, FRAMES, 84, 84)
        x = (x / 255.0)  # rescale pixel value as 0 to 1.
        F = self.conv(x)
        V = self.value_stream(F)
        A = self.advantage_stream(F)
        Q = V + (A - A.mean())
        return Q


class ReplayMemory:
    '''FIFO buffer for storing experience tuples.'''

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.idx = 0

    def add(self, item):
        '''Add an item to the buffer.'''
        if len(self.memory) < self.size:
            self.memory.append(item)
        else:
            self.memory[self.idx] = item
        self.idx = (self.idx + 1) % self.size  # circulur buffer

    def sample(self, n):
        '''Randomly sample n elements from the buffer'''
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


class Agent:

    def __init__(self, env):
        self.env = env
        self.states_n = self.env.observation_space.shape[0]
        self.actions_n = self.env.action_space.n
        self.model = Model3Layer(self.actions_n, lr=LR).to(device)
        self.target_model = Model3Layer(self.actions_n, lr=LR).to(device)
        self.target_model.load_state_dict(self.model.state_dict())  # copy weights to target model
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        #print(summary(self.model, (4, 84, 84)))  # show summary of model archicture
        print(f'Model: {self.model.__class__}')

    def _act(self, state, eps):
        '''Given a state return an action following an epsilon greedy policy.'''
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            state = np.array(state)  # convert LazyFrames (list of arrays) to single array for better performance
            state = np.squeeze(state)  # remove extra dim added by ResizeObservation wrapper
            state = np.expand_dims(state, 0)  # make a mini-batch of 1
            return np.argmax(self.model.forward(torch.tensor(state).to(device)).detach().cpu().numpy())

    def _learn(self):
        '''Sample experience from replay memory.  Generate targets and predictions.  Update the model.'''
        experience = self.replay_memory.sample(BATCH_SIZE)  # randomly sample from the buffer
        states, actions, rewards, next_states, dones = zip(*experience)  # unzip experience tuple into separate tuples

        # convert to tensors
        states = torch.tensor(np.stack(states, 0)).float().to(device)  # stack tuple into mini-batch, adding dim 0
        actions = torch.tensor(actions).long().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        next_states = torch.tensor(np.stack(next_states, 0)).float().to(device)  # stack tuple into mini-batch, adding dim 0
        dones = torch.tensor(dones).long().to(device)
        # remove extra dim added by ResizeObservation wrapper
        states = states.squeeze()
        next_states = next_states.squeeze()
        # sanity checks
        assert states.shape == next_states.shape == (BATCH_SIZE, FRAMES, 84, 84)
        assert actions.shape == rewards.shape == dones.shape == (BATCH_SIZE,)

        # calculate DQN target
        #Q_target = self.target_model.forward(next_states)  # predict value of next_state using target model)
        #assert Q_target.shape == (BATCH_SIZE, self.actions_n)
        #targets = rewards + (GAMMA * Q_target.detach().max(1)[0]) * (1 - dones)  # take max (greedy action).  (1 - dones) makes target = reward for terminal states
        #assert targets.shape == (BATCH_SIZE,)

        #calculate DDQN target (used to reduce maximization bias)
        Q = self.model.forward(next_states).detach()  # predict value of next_state state using model
        assert Q.shape == (BATCH_SIZE, self.actions_n)
        actions_idx = Q.argmax(1)  # get the indices of the best actions
        assert actions_idx.shape == (BATCH_SIZE,)
        Q_target = self.target_model.forward(next_states).detach()  # predict value of next_state state using target model
        assert Q_target.shape == (BATCH_SIZE, self.actions_n)
        targets = rewards + (GAMMA * Q_target.gather(1, actions_idx.unsqueeze(1)).squeeze()) * (1 - dones)  # use action indicies to select action from target model
        assert targets.shape == (BATCH_SIZE,)

        # calculate prediction
        Q_pred = self.model.forward(states)  # predict value of current state using model
        predictions = Q_pred.gather(1, actions.unsqueeze(1)).squeeze()  # prediction is based on action taken
        assert predictions.shape == (BATCH_SIZE,)

        # calculate loss
        assert predictions.shape == targets.shape
        loss = nn.functional.huber_loss(predictions, targets)

        # backprop
        self.model.optimizer.zero_grad()  # Sets gradients of all model parameters to zero.
        loss.backward()  # Computes the gradient of current tensor w.r.t. graph leaves.
        self.model.optimizer.step()  # Performs a single optimization step.
        return loss.detach().cpu().numpy()

    def train(self, filename=None):
        # optionally load saved model and associated replay memory to continue training
        if filename:
            print(f'Loading: {filename}, setting epsilon to {EPS_MIN}')
            global EPS_START
            EPS_START = EPS_MIN
            self.model.load_state_dict(torch.load(f'{filename}', map_location=torch.device('cpu')))
            self.target_model.load_state_dict(torch.load(f'{filename}', map_location=torch.device('cpu')))
            filename = Path(filename)
            with open(filename.with_suffix('.pkl'), 'rb') as file:
                self.replay_memory = pickle.load(file)

        # create save paths
        training_run_path = Path('training_runs/' + DEVICE + '-' + str(datetime.now()).replace(' ', '-'))  # unique folder per training run
        training_run_path.mkdir(parents=True)
        models_path = training_run_path / 'models'  # models
        models_path.mkdir()
        runs_path = training_run_path / 'runs'  # tensorboard logging
        logging.basicConfig(filename=training_run_path / 'train.log', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
        print(f'Path: {training_run_path}')

        rewards = []  # total reward per episode
        train_steps = 0  # number of steps taken over entire training run
        n = 0  # episode count
        total_run_time = 0
        writer = SummaryWriter(log_dir=runs_path)

        while train_steps <= TRAIN_STEPS_MAX:
            n += 1
            logging.debug(f'-------------------------- STARTING EPISODE {n} --------------------------')
            t0 = time.time()
            # initialize counters
            episode_reward = 0
            episode_steps = 0
            episode_loss = 0
            episode_act_time = 0
            episode_environment_time = 0
            episode_learn_time = 0
            eps = EPS_START
            # initialize environment
            state = self.env.reset()
            done = False
            while not done:
                # choose action
                t0_act = time.time()
                action = self._act(state, eps)  # take an action using e-greedy policy
                t1_act = time.time()
                episode_act_time += (t1_act - t0_act)

                # take action
                t0_env = time.time()
                next_state, reward, done, info = self.env.step(action)  # step the environment

                # reward hacking
                #if done:
                #    if info['flag_get']:  # if mario reaches end of the level
                #        reward += 500.0
                logging.debug(f'Episode {n}, step {episode_steps}.  Took action {action}, received {round(reward, 2)} reward, done is {done}, info is {info}.')
                t1_env = time.time()
                episode_environment_time += (t1_env - t0_env)

                # add to replay memory
                self.replay_memory.add(tuple([state, action, reward, next_state, done]))  # add [s, a, r, s'] to replay memory

                # learn
                if (train_steps >= REPLAY_MEMORY_MIN) and (train_steps % LEARN_EVERY == 0):  # once replay memory has accumulated some experience
                    t0_learn = time.time()
                    logging.debug('Learning')
                    loss = self._learn()
                    episode_loss += loss  # accumulate loss
                    t1_learn = time.time()
                    episode_learn_time += (t1_learn - t0_learn)

                # sync target model
                if (train_steps % SYNC_TARGET_MODEL_EVERY == 0) and (train_steps != 0):  # every SYNC_TARGET_MODEL_EVERY steps
                    logging.debug('Syncing target model')
                    self.target_model.load_state_dict(self.model.state_dict())  # copy weights to target model

                episode_reward += reward  # accumulate reward
                episode_steps += 1  # increment episode step count
                train_steps += 1  # increment training step count
                state = next_state  # move to next state
                eps = EPS_START - ((EPS_START - EPS_MIN) / EPS_DECAY_STEPS) * train_steps  # decay epsilon
                eps = max(EPS_MIN, eps)
            rewards.append(episode_reward)
            smoothed_rewards = moving_average(rewards)
            t1 = time.time()
            episode_run_time = t1 - t0
            total_run_time += episode_run_time
            steps_per_second = episode_steps / episode_run_time

            # log to tensorboard
            writer.add_scalar("Epsilon", eps, train_steps)
            writer.add_scalar("Reward", episode_reward, train_steps)
            writer.add_scalar("Loss", episode_loss, train_steps)
            writer.add_scalar("Replay memory used", len(self.replay_memory), train_steps)
            writer.add_scalar("Steps per second", steps_per_second, train_steps)
            writer.add_scalar("Total run time", total_run_time, n)
            writer.add_scalar("Episode steps", episode_steps, n)
            writer.add_scalar("Episode act time", (episode_act_time), n)
            writer.add_scalar("Episode environment time", (episode_environment_time), n)
            writer.add_scalar("Episode learn time", (episode_learn_time), n)
            writer.add_scalar("Episode run time", episode_run_time, n)

            # show and save intermediate results
            if (n % SHOW_PROGRESS_EVERY == 0):
                print(
                    f'Ep: {n}\tAvgR: {round(smoothed_rewards[-1], 2)}\tBestAvgR: {round(np.max(smoothed_rewards), 2)}\tEps: {round(eps, 4)}\tRepBuf: {len(self.replay_memory)}\tSteps:{episode_steps}\tTotSteps: {train_steps}\tRunTime: {round(episode_run_time)}s ({round(episode_act_time)}/{round(episode_environment_time)}/{round(episode_learn_time)})\tTotRunTime: {round(total_run_time)}s\tSteps/s: {round(steps_per_second)}'
                )
            if n % SAVE_MODEL_EVERY == 0:
                torch.save(self.model.state_dict(), models_path / f'episode_{n}.pth')
                torch.save(self.model.state_dict(), models_path / 'latest.pth')
                with open(models_path / 'latest.pkl', 'wb') as file:
                    pickle.dump(self.replay_memory, file)
            if episode_reward >= max(rewards):
                torch.save(self.model.state_dict(), models_path / 'best.pth')
        torch.save(self.model.state_dict(), models_path / 'final.pth')
        with open(models_path / 'final.pkl', 'wb') as file:
            pickle.dump(self.replay_memory, file)
        writer.flush()
        writer.close()

    def eval(self, episodes=10, epsilon=0.01, filename=None, render=False):
        'Evaluate trained agent.'
        self.model.load_state_dict(torch.load(f'{filename}', map_location=torch.device('cpu')))

        rewards = []  # total reward per episode
        for n in range(episodes):
            t0 = time.time()
            state = self.env.reset()
            if render == 'video':
                self.env.start_video_recorder()
            done = False
            episode_reward = 0
            episode_steps = 0
            self.model.eval()
            with torch.no_grad():
                while not done:
                    if render == 'human':
                        self.env.render()
                        time.sleep(FRAMESKIP / 100)
                    action = self._act(state, epsilon)  # take an action using e-greedy policy
                    state, reward, done, info = self.env.step(action)  # step the environment
                    episode_reward += reward  # accumulate reward
                    episode_steps += 1  # increment step count
            rewards.append(episode_reward)
            t1 = time.time()
            print(f'Agent ran for {episode_steps} steps, received {round(episode_reward, 2)} reward.  RunTime: {round(t1 - t0)}s')
        print()
        print(f'Average Episode Reward across {episodes} episodes: {np.mean(rewards)}')


def moving_average(a, n=MOVING_AVERAGE):
    # don't return results until we've collected enough data
    if len(a) < n:
        return [-np.inf]
    else:
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


def pre_process_env(env):
    env = NoopResetEnv(env)
    env = SkipFrame(env, skip=FRAMESKIP)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, FRAMES)
    return env


if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str)  # --mode: ('train', 'eval)
    parser.add_argument('-f', type=str)  # --filename: model filename
    parser.add_argument('-r', type=str)  # --render: ('human', 'video')
    parser.add_argument('-s', type=int)  # --steps: number of steps to run
    args = parser.parse_args()

    # set pytorch num threads and device
    if PYTORCH_THREADS:
        torch.set_num_threads(PYTORCH_THREADS)
    print('Pytorch threads:', torch.get_num_threads())
    print('Pytorch device:', DEVICE)
    device = torch.device(DEVICE)

    def train():
        gym.logger.set_level(gym.logger.ERROR)
        env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=['1-1', '1-2', '1-3', '1-4'])
        env = JoypadSpace(env, RIGHT_ONLY)
        #env = gym.make(ENV, continuous=False)
        env = pre_process_env(env)
        agent = Agent(env)
        agent.train(filename=args.f)
        env.close()

    def evaluate():
        #gym.logger.set_level(gym.logger.ERROR)
        env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=['1-1', '1-2', '1-3', '1-4'])
        env = JoypadSpace(env, RIGHT_ONLY)
        #env = gym.make(ENV, continuous=False)
        env = pre_process_env(env)
        if args.s:
            env._max_episode_steps = args.s
        if args.r == 'video':
            # create save paths
            eval_videos_path = Path('eval_videos/' + str(datetime.now()).replace(' ', '-'))  # unique folder per eval run
            print(f'Videos path: {eval_videos_path}')
            env = RecordVideo(env, eval_videos_path)
        agent = Agent(env)
        agent.eval(episodes=10, epsilon=0.01, filename=args.f, render=args.r)
        env.close()

    if args.m == 'train':
        train()
    elif args.m == 'eval':
        evaluate()
