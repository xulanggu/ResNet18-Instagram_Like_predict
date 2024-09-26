import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class InstagramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        incorrect_img_path = self.data.iloc[idx, 4]
        
        corrected_img_path = incorrect_img_path.replace('../Data/insta_data/', 'insta_data/')
        

        image = Image.open(corrected_img_path).convert('RGB')
        likes = self.data.iloc[idx, 0]
        print(likes)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(likes, dtype=torch.float32)


