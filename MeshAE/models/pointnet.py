import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


class PointNetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(PointNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(1024, embedding_size)

    def forward(self, x):
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc_mu(x)

        return x


class PointNetDecoder(nn.Module):
    def __init__(self, embedding_size, output_channels=3, num_points=1024):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.output_channels = output_channels
        self.fc1 = nn.Linear(embedding_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 4096)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(4096)
        self.fc4 = nn.Linear(4096, num_points * output_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.leaky_relu(self.bn1(self.fc1(x)),0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)),0.2)
        x = self.fc4(x)
        x = x.view(batch_size, self.num_points, self.output_channels)
        x = x.contiguous()
        return x


class PointNetAE(nn.Module):
    def __init__(self, embedding_size=256, input_channels=6, output_channels=6, num_points=1024, normalize=True):
        super(PointNetAE, self).__init__()
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = embedding_size
        self.encoder = PointNetEncoder(embedding_size, input_channels)
        self.decoder = PointNetDecoder(embedding_size, output_channels, num_points)

    def encode(self, x):
        z = self.encoder(x)
        if self.normalize:
            z = F.normalize(z)
        return z

    def decode(self, z):
        y = self.decoder(z)
        y1,y2 = torch.split(y,3,dim=2)
        y2 =y2/torch.sqrt(torch.sum(y2**2,dim=2,keepdim=True))
        y = torch.concatenate([y1,y2],dim=2)
        return y
