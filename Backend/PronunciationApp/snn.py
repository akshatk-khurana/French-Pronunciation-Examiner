"""
This module defines the Siamese Neural Network and audio preprocessing pipeline for the 
audio files collected.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import librosa
import numpy as np

def pad_or_trim(mel_tensor, target_width):
    """Pad or trim the mel spectrogram tensor to a fixed time dimension.

    Args:
        mel_tensor: The mel spectrogram tensor of shape (1, 128, time_steps).
        target_width: The target number of time steps.
    
    Returns:
        mel_tensor: The padded or trimmed to (1, 1, 128, target_width).
    """
    _, mel_bands, time_steps = mel_tensor.shape
    if time_steps > target_width:
        mel_tensor = mel_tensor[:, :, :target_width]
    elif time_steps < target_width:
        pad_amount = target_width - time_steps
        mel_tensor = F.pad(mel_tensor, (0, pad_amount))
    return mel_tensor

def load_and_process(path):
    """Load an audio file, trim silence, compute mel spectrogram, normalize, and pad/trim.

    Args:
        path: A string path to the audio file.
    
    Returns:
        mel_tensor: The processed mel spectrogram tensor ready for the model.
    """
    y, sr = librosa.load(path, sr=22050)
    y, _ = librosa.effects.trim(y, top_db=15 )

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=1024)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.min()) / (mel.max() - mel.min())

    mel_tensor = torch.tensor(mel).unsqueeze(0) # (1, 1, 128, time_steps)

    mel_tensor = pad_or_trim(mel_tensor, 64)

    return mel_tensor

class SiameseNetwork(nn.Module):
    """A Siamese neural network for comparing two audio samples (the spoken and standard) 
    using convolutional layers."""
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
        """Extract a flattened feature vector from the input tensor using 
        the convolutional layers."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, std, spk):
        """Compare two processed audio tensors and return similarity score."""
        std_feature_vector = self.get_feature_vector(std)
        spk_feature_vector = self.get_feature_vector(spk)

        concatenated = torch.cat((std_feature_vector, spk_feature_vector), 1)
        features = self.feature_similarity(concatenated)
        similarity = self.sigmoid(features)
        return similarity