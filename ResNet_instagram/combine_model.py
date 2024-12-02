import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from instagram_dataset_v2 import InstagramDataset

class CombinedModel(nn.Module):
    def __init__(self, num_numerical_features):
        super(CombinedModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  

        self.fc_numerical = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fc_combined = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  
        )

    def forward(self, image, numerical_features):
        image_features = self.resnet(image)
        numerical_features = self.fc_numerical(numerical_features)
        combined_features = torch.cat((image_features, numerical_features), dim=1)
        output = self.fc_combined(combined_features)
        return output
