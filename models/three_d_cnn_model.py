# models/3d_cnn_model.py

import torch
import torch.nn as nn
import torchvision

class R3DViolenceDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(R3DViolenceDetector, self).__init__()
        # Load pretrained 3D ResNet18
        self.r3d = torchvision.models.video.r3d_18(pretrained=pretrained)
        self.r3d.fc = nn.Linear(self.r3d.fc.in_features, 1)  # Binary output

    def forward(self, x):
        return self.r3d(x).squeeze(1)

# Example usage in train.py:
# from models.3d_cnn_model import R3DViolenceDetector
# model = R3DViolenceDetector(pretrained=True).to(device)
# Clip shape should be [B, C, T, H, W] (e.g., [4, 3, 16, 112, 112])

# Note: The dataset loader must stack frames as [C, T, H, W] per clip instead of [T, C, H, W].
