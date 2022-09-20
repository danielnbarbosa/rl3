import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT


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


def preprocess_env(env, frameskip, frames):
    env = NoopResetEnv(env)
    env = SkipFrame(env, skip=frameskip)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = gym.wrappers.gray_scale_observation.GrayScaleObservation(env)  # convert to grayscale
    env = gym.wrappers.resize_observation.ResizeObservation(env, 84)  # resize to (84,84)
    env = gym.wrappers.frame_stack.FrameStack(env, frames)
    return env
