import torch.nn as nn
import torch.optim as optim


class Model2Layer(nn.Module):
    """
    Input is 4 stacked 84x84 int8 grayscale frames.
    This is the architecture from Deepmind 2013 paper "Playing Atari with Deep Reinforcement Learning".
    Added dueling networks which doubles the number of params.
    Total params: 1.3M
    """

    def __init__(self, outputs, lr):
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
        self.optimizer = optim.Adam(self.parameters(), lr, eps=1e-4)

    def forward(self, x):
        assert x.shape == (1, 4, 84, 84) or x.shape == (32, 4, 84, 84)
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

    def __init__(self, outputs, lr):
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
        self.optimizer = optim.Adam(self.parameters(), lr, eps=1e-4)

    def forward(self, x):
        assert x.shape == (1, 4, 84, 84) or x.shape == (32, 4, 84, 84)
        x = (x / 255.0)  # rescale pixel value as 0 to 1.
        F = self.conv(x)
        V = self.value_stream(F)
        A = self.advantage_stream(F)
        Q = V + (A - A.mean())
        return Q
