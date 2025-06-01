import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(SmallCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # stack_size

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))
