
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class PointNetAbstract(nn.Module):
    def __init__(self):
        super(PointNetAbstract, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)

        # Add a fully connected layer
        self.f1 = nn.Linear(512 * 16, 2048)  # Reduce 8192 to 2048 
        self.f2 = nn.Linear(2048, 1024)  # Reduce 2048 to 1024 
        self.f3 = nn.Linear(1024, 512)  # Reduce 1024 to 512 
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Flatten the output
        flatl4_points = l4_points.view(l4_points.size(0), -1)  # (batch_size, 8192)
        
        # Apply the fully connected layer, batch normalization, and activation function
        fc_points = self.f1(flatl4_points)
        fc_points = self.f2(fc_points)
        fc_points = self.f3(fc_points)
        
        if fc_points.shape[0] > 1:
            bn1_points = self.bn1(fc_points)  # Apply batch normalization
        else:
            bn1_points = fc_points
        fpoints = F.relu(bn1_points)  # Activation function (ReLU)

        return fpoints