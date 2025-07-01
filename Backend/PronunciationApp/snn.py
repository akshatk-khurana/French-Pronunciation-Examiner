import torch.nn as nn
import torch
import torch.nn.functional as F
import librosa
import numpy as np

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

    self.feature_similarity = nn.Linear(6688, 1)

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

def pad_or_trim(mel_tensor, target_width):
    _, mel_bands, time_steps = mel_tensor.shape
    if time_steps > target_width:
        mel_tensor = mel_tensor[:, :, :target_width]
    elif time_steps < target_width:
        pad_amount = target_width - time_steps
        mel_tensor = F.pad(mel_tensor, (0, pad_amount))
    return mel_tensor

def load_and_process(path):
    y, sr = librosa.load(path, sr=22050)
    y, _ = librosa.effects.trim(y, top_db=15 )

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=1024)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.min()) / (mel.max() - mel.min())

    mel_tensor = torch.tensor(mel).unsqueeze(0) # (1, 1, 128, time)

    mel_tensor = pad_or_trim(mel_tensor, 64)

    return mel_tensor