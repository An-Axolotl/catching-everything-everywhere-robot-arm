import torch
import torch.nn as nn
import sys
import torchvision
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
#from torchvision.models.resnet import resnet18
from timm.models.convnext import convnextv2_nano, convnextv2_atto, convnextv2_femto

class EmptyDepthBackbone(nn.Module):
    def __init__(self, output_latent_size):
        super().__init__()

        self.output_latent_size = output_latent_size

    def forward(self, image, obs_buf):
        num_envs = obs_buf.shape[0]
        device = obs_buf.device

        return torch.zeros((num_envs, self.output_latent_size), device=device)

class ConvnextBackbone(nn.Module):
    def __init__(self, num_cameras, num_input, latent_dim, res=[120, 160]):
        super().__init__()
        height, width = res
        assert(num_cameras == 1)
        assert(height % 4 == 0 and width % 4 == 0)
        self.latent_dim = latent_dim
        self.backbone = convnextv2_atto(in_chans=num_input, num_classes=latent_dim, pretrained=False)

    def forward(self, images, obs_input=None):
        features = self.backbone(images)
        return features 

class ResnetBackbone(nn.Module):
    def __init__(self, num_cameras, num_input, latent_dim, res=[120, 160], **kwargs):
        super().__init__()
        height, width = res
        assert(num_cameras == 1)
        assert(height % 4 == 0 and width % 4 == 0)

        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(num_input, 32, kernel_size=5, padding=2), # [1, h, w] -> [32, h/2, w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [32, h/2, w/2] -> [128, h/8, w/8]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [32, h/2, w/2] -> [128, h/8, w/8]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(), # [2*h*w]
            nn.Linear(128 * (height//8) * (width // 8), 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, images, obs_input=None):
        features = self.backbone(images)
        return features 


class RecurrentResnetDepthBackbone(nn.Module):
    def __init__(self, num_cameras, num_input, latent_dim, num_obs_input, res, **kwargs):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = ResnetBackbone(num_cameras, num_input, 64, res)
        self.combination_mlp = nn.Sequential(
            nn.Linear(64+num_obs_input, 128),
            activation,
            nn.Linear(128, 64)
        )
        self.rnn = nn.GRU(input_size=64, hidden_size=512, batch_first=True, num_layers=1)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, latent_dim),
            #last_activation,
            #nn.Linear(128, latent_dim)
        )
        self.hidden_states = None
        self.is_rnn = True

    def forward(self, depth_image, obs_input):
        depth_image = self.base_backbone(depth_image)
        depth_latent = torch.cat((depth_image, obs_input), dim=-1)
        depth_latent = self.combination_mlp(depth_latent)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent


class RecurrentResnetDepthBackbone2(nn.Module):
    def __init__(self, num_cameras, num_input, latent_dim, num_obs_input, res, **kwargs):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = ResnetBackbone(num_cameras, num_input, 128, res)
        self.combination_mlp = nn.Sequential(
            nn.Linear(128+num_obs_input, 256),
            activation,
            nn.Linear(256, 128)
        )
        self.rnn = nn.GRU(input_size=128, hidden_size=1024, batch_first=True, num_layers=1)
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, latent_dim),
            #last_activation,
            #nn.Linear(128, latent_dim)
        )
        self.hidden_states = None
        self.is_rnn = True

    def forward(self, depth_image, obs_input):
        depth_image = self.base_backbone(depth_image)
        depth_latent = torch.cat((depth_image, obs_input), dim=-1)
        depth_latent = self.combination_mlp(depth_latent)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

class ResnetBackbone2(nn.Module):
    def __init__(self, num_cameras, num_input, latent_dim, res=[120, 160], **kwargs):
        super().__init__()
        height, width = res
        assert(num_cameras == 1)
        assert(height % 4 == 0 and width % 4 == 0)

        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(num_input, 64, kernel_size=5, padding=2), # [1, h, w] -> [32, h/2, w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [32, h/2, w/2] -> [128, h/8, w/8]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(), # [2*h*w]
            nn.Linear(128 * (height//4) * (width // 4), 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, images, obs_input=None):
        features = self.backbone(images)
        return features 