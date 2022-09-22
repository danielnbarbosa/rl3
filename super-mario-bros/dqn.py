'''
DQN for learning from pixels.
Converts RGB observation to scaled, grayscale stack of frames.
In addition to vanilla DQN, this also implements dueling networks and double DQN.

Customizations specific to super-mario-bros:
- imports for gym_super_mario_bros
- JoypadSpace wrapper to limit available actions
- make environment using gym_super_mario_bros.make()
- training path uses ENV instead of ENV.split()
- in NoopResetEnv wrapper, noops = self.unwrapped.np_random.randint instead of self.unwrapped.np_random.integers
- using SIMPLE_MOVEMENT action space
- disabled NoopReset wrapper
- added CustomReward wrapper
- added old style env.render() to render game for human
- added old style video recorder used in gym 0.23.0
- made evaluate use epsilon greedy during training and greedy during final eval
- evaluate more frequently
- reduce EPS_DECAY_STEPS, REPLAY_MEMORY_MIN, REPLAY_MEMORY_SIZE by 1/4
- set GAMMA to 0.9
'''

import logging
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import gym
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from models import Model3Layer
from wrappers import preprocess_env
from memory import ReplayMemory
import gym_super_mario_bros
from old_video_recorder.monitor import Monitor

#ENV = 'SuperMarioBros-1-1-v0'
#ENV = 'SuperMarioBros-1-2-v0'
ENV = 'SuperMarioBros-v0'

TRAIN_STEPS_MAX = 50_000_000  # train for this many steps, will go a little beyond to finish the current episode
REPLAY_MEMORY_MIN = 50_000  # minimum amount of accumulated experience before before we begin sampling
REPLAY_MEMORY_SIZE = 250_000  # max size of replay memory buffer
BATCH_SIZE = 32  # number of items to randomly sample from replay memory
SYNC_TARGET_MODEL_EVERY = 10_000  # how often (in steps) to copy weights to target model
LEARN_EVERY = 4  # update model weights every n steps via gradient descent
FRAMES = 4  # number of observations to stack together to form the state
FRAMESKIP = 4  # number of frames to repeat the same actions
LR = 0.00025  # learning rate
GAMMA = 0.9  # discount rate
EPS_START = 1  # starting value of epsilon
EPS_MIN = .1  # minimum value for epsilon
EPS_DECAY_STEPS = 250_000  # over how many steps to linearly reduce epsilon until it reaches EPS_MIN
EVAL_MODEL_EVERY = 100_000  # how often (in steps) to evaluate the model


class Agent:

    def __init__(self, env):
        self.env = env
        self.states_n = self.env.observation_space.shape[0]
        self.actions_n = self.env.action_space.n
        self.model = Model3Layer(self.actions_n, LR).to(device)
        self.target_model = Model3Layer(self.actions_n, LR).to(device)
        self.target_model.load_state_dict(self.model.state_dict())  # copy weights to target model
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        #print(summary(self.model, (4, 84, 84)))  # show summary of model archicture

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
        training_run_path = Path('training_runs/' + device.type + '-' + ENV + '-' + str(datetime.now()).replace(' ', '-'))  # unique folder per training run
        training_run_path.mkdir(parents=True)
        models_path = training_run_path / 'models'  # models
        models_path.mkdir()
        runs_path = training_run_path / 'runs'  # tensorboard logging
        # set level to logging.DEBUG to enable debug logging, set to logging.ERROR to disable
        logging.basicConfig(filename=training_run_path / 'train.log', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.ERROR, datefmt='%Y-%m-%d %H:%M:%S')
        print(f'Path: {training_run_path}')

        train_steps = 0  # number of steps taken over entire training run
        n = 0  # episode count
        total_run_time = 0  # total time training
        eval_reward = 0  # average reward achieved during evaluation
        best_eval_reward = 0
        writer = SummaryWriter(log_dir=runs_path)

        while train_steps <= TRAIN_STEPS_MAX:
            n += 1
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
                t1_env = time.time()
                episode_environment_time += (t1_env - t0_env)

                # add to replay memory
                self.replay_memory.add(tuple([state, action, reward, next_state, done]))  # add [s, a, r, s'] to replay memory

                # learn
                if (train_steps >= REPLAY_MEMORY_MIN) and (train_steps % LEARN_EVERY == 0):  # once replay memory has accumulated some experience
                    t0_learn = time.time()
                    loss = self._learn()
                    episode_loss += loss  # accumulate loss
                    t1_learn = time.time()
                    episode_learn_time += (t1_learn - t0_learn)

                # sync target model
                if (train_steps % SYNC_TARGET_MODEL_EVERY == 0) and (train_steps != 0):  # every SYNC_TARGET_MODEL_EVERY steps
                    self.target_model.load_state_dict(self.model.state_dict())  # copy weights to target model

                # evaluate
                if (train_steps % EVAL_MODEL_EVERY == 0) and (train_steps != 0):
                    # save intermediate models
                    torch.save(self.model.state_dict(), models_path / f'train_steps_{train_steps}.pth')
                    torch.save(self.model.state_dict(), models_path / 'latest.pth')
                    with open(models_path / 'latest.pkl', 'wb') as file:
                        pickle.dump(self.replay_memory, file)

                    eval_reward = evaluate(models_path / 'latest.pth')
                    if eval_reward > best_eval_reward:
                        print(f'Saving new best model with eval_reward of {eval_reward}')
                        torch.save(self.model.state_dict(), models_path / 'best.pth')
                        best_eval_reward = eval_reward

                episode_reward += reward  # accumulate reward
                episode_steps += 1  # increment episode step count
                train_steps += 1  # increment training run step count
                state = next_state  # move to next state
                eps = EPS_START - ((EPS_START - EPS_MIN) / EPS_DECAY_STEPS) * train_steps  # decay epsilon
                eps = max(EPS_MIN, eps)
            t1 = time.time()
            episode_run_time = t1 - t0
            total_run_time += episode_run_time
            steps_per_second = episode_steps / episode_run_time

            # log to tensorboard
            writer.add_scalar("Epsilon", eps, train_steps)
            writer.add_scalar("Reward Train", episode_reward, train_steps)
            writer.add_scalar("Reward Eval", eval_reward, train_steps)
            writer.add_scalar("Loss", episode_loss, train_steps)
            writer.add_scalar("Replay memory used", len(self.replay_memory), train_steps)
            writer.add_scalar("Steps per second", steps_per_second, train_steps)
            writer.add_scalar("Total run time", total_run_time, n)
            writer.add_scalar("Episode steps", episode_steps, n)
            writer.add_scalar("Episode act time", (episode_act_time), n)
            writer.add_scalar("Episode environment time", (episode_environment_time), n)
            writer.add_scalar("Episode learn time", (episode_learn_time), n)
            writer.add_scalar("Episode run time", episode_run_time, n)

            # show intermediate results
            print(
                f'Ep: {n}\tReward: {round(episode_reward,2)}\tEps: {round(eps, 4)}\tBufLen: {len(self.replay_memory)}\tSteps:{episode_steps}\tTotSteps: {train_steps}\tRunTime: {round(episode_run_time)}s ({round(episode_act_time)}/{round(episode_environment_time)}/{round(episode_learn_time)})\tTotRunTime: {round(total_run_time)}s\tSteps/s: {round(steps_per_second)}\tEvalReward: {round(eval_reward, 2)}'
            )
        # save final model
        torch.save(self.model.state_dict(), models_path / 'final.pth')
        with open(models_path / 'final.pkl', 'wb') as file:
            pickle.dump(self.replay_memory, file)
        writer.flush()
        writer.close()


def evaluate(filename, episodes=30, epsilon=0.05, render_mode=None):
    'Evaluate trained model.  Uses fresh env and agent to avoid interacting with training.'

    print(f'Evaluating {filename}')
    # instantiate new gym environment and agent
    gym.logger.set_level(gym.logger.ERROR)
    #env = gym.make(ENV, full_action_space=False, frameskip=1, repeat_action_probability=0.25, render_mode=render_mode)
    env = gym_super_mario_bros.make(ENV)
    env = preprocess_env(env, FRAMESKIP, FRAMES)

    if render_mode == 'rgb_array':
        # create save path
        eval_videos_path = Path('eval_videos/' + str(datetime.now()).replace(' ', '-'))  # unique folder per eval run
        print(f'Videos path: {eval_videos_path}')
        #env = gym.wrappers.record_video.RecordVideo(env, eval_videos_path, episode_trigger=lambda x: True)
        env = Monitor(env, eval_videos_path, video_callable=lambda episode_id: True, force=True)

    agent = Agent(env)
    agent.model.load_state_dict(torch.load(f'{filename}', map_location=torch.device('cpu')))

    rewards = []  # total reward per episode
    for n in range(episodes):
        t0 = time.time()
        state = agent.env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        agent.model.eval()
        with torch.no_grad():
            while not done:
                if render_mode == 'human':
                    env.render()
                    time.sleep(FRAMESKIP / 100)
                action = agent._act(state, epsilon)  # take an action using e-greedy policy
                state, reward, done, info = agent.env.step(action)  # step the environment
                episode_reward += reward  # accumulate reward
                episode_steps += 1  # increment step count
        rewards.append(episode_reward)
        t1 = time.time()
        print(f'Run {n}, agent ran for {episode_steps} steps, received {round(episode_reward, 2)} reward.  RunTime: {round(t1 - t0)}s')
    # cleanup
    env.close()
    del agent
    # return results
    mean_reward = np.mean(rewards)
    print()
    print(f'Average episode reward across {episodes} episodes: {round(mean_reward, 2)}.  Best reward: {round(max(rewards), 2)}')
    return mean_reward


if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str)  # mode: ('train', 'eval)
    parser.add_argument('-f', type=str)  # filename: model filename
    parser.add_argument('-r', type=str)  # render: ('human', 'rgb_array')
    args = parser.parse_args()

    # show pytorch num threads and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Pytorch threads:', torch.get_num_threads())
    print('Pytorch device:', device)

    if args.m == 'train':
        gym.logger.set_level(gym.logger.ERROR)
        #env = gym.make(ENV, full_action_space=False, frameskip=1, repeat_action_probability=0.25)
        env = gym_super_mario_bros.make(ENV)
        env = preprocess_env(env, FRAMESKIP, FRAMES)
        agent = Agent(env)
        agent.train(filename=args.f)
        env.close()
    elif args.m == 'eval':
        evaluate(args.f, epsilon=0.05, render_mode=args.r)
