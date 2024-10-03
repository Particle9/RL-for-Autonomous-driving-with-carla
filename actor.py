import torch
import torch.nn as nn
import torch.nn.functional as F

from customTransformer import TransformerBlock
from imageFeatureExtractor import ResNetFeatureExtractor
from pointCloudFeatureExtractor import PointNetAbstract

class Actor(nn.Module):
    def __init__(self, image_shape = (480,640), pointnet_weights = None):
        super(Actor,self).__init__()
        self.imagenet= ResNetFeatureExtractor(input_size = image_shape)
        self.pointnet = PointNetAbstract()

        if pointnet_weights:
            self.pointnet.load_state_dict(pointnet_weights, strict=False)

        self.transformer = TransformerBlock(in_dim=1024, num_heads=8, ff_dim=512, out_dim=3)

    def forward(self, image_input, point_input):
        image_feature = self.imagenet(image_input)
        point_feature = self.pointnet(point_input)

        features = torch.cat([image_feature,point_feature], dim=-1)

        output = self.transformer(features)

        # Apply specific activation functions for throttle, brake, and steering
        throttle = torch.tanh(output[:, 0])  # Throttle can be negative or positive
        brake = torch.relu(output[:, 1])     # Brake cannot be negative
        steering = torch.tanh(output[:, 2])  # Steering can be negative or positive

        # Stack the outputs back together
        actions = torch.stack((throttle, brake, steering), dim=1)

        
        return actions

    def preprocess_image(self, image):
        return self.imagenet.preprocess_image(image)    