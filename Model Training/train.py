import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.utils.data

class SiameseNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=3
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=3
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=3
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.sigmoid = nn.Sigmoid()

    self.feature_similarity = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(6688, 1),
    )

    self.sigmoid = nn.Sigmoid()

  def get_feature_vector(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)

      x = torch.flatten(x, 1)
      return x


  def forward(self, std, spk):
    std_feature_vector = self.get_feature_vector(std)

    spk_feature_vector = self.get_feature_vector(spk)

    concatenated = torch.cat((std_feature_vector, spk_feature_vector), 1)

    features = self.feature_similarity(concatenated)

    similarity = self.sigmoid(features)

    return similarity