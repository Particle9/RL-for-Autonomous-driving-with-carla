import torch
import torch.nn as nn
import torch.nn.functional as F

from customTransformer import TransformerBlock
from imageFeatureExtractor import ResNetFeatureExtractor
from pointCloudFeatureExtractor import PointNetAbstract


class Critic(nn.Module):
    def __init__(self, image_shape=(480, 640), pointnet_weights=None):
        super(Critic, self).__init__()
        # Image feature extraction using ResNet
        self.imagenet = ResNetFeatureExtractor(input_size=image_shape)
        # Point cloud feature extraction using PointNet
        self.pointnet = PointNetAbstract()

        if pointnet_weights:
            self.pointnet.load_state_dict(pointnet_weights, strict=False)

        # Transformer block to process the concatenated features and action
        self.transformer = TransformerBlock(in_dim=1024 + 3, num_heads=8, ff_dim=512, out_dim=1)

    def forward(self, image_input, point_input, action_input):
        # Extract image and point cloud features
        image_feature = self.imagenet(image_input)
        point_feature = self.pointnet(point_input)

        # Concatenate the image and point cloud features
        features = torch.cat([image_feature, point_feature], dim=-1)

        # Concatenate the action to the features
        combined_input = torch.cat([features, action_input], dim=-1)

        # Process through the transformer
        q_value = self.transformer(combined_input)

        # Output the estimated Q-value
        return q_value
    
    def preprocess_image(self, image):
        return self.imagenet.preprocess_image(image)    