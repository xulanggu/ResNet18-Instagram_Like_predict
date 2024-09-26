import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import os

csv_file = 'instagram_data.csv'
df = pd.read_csv(csv_file)


print(df.head())

from instagram_dataset import InstagramDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = InstagramDataset(csv_file='instagram_data.csv', root_dir='insta_data', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)



for images, likes in train_loader:
    print(f'Batch of train_images has shape: {images.shape}')
    print(f'Batch of train_likes: {likes}')
    
    break  
for images, likes in test_loader:
    print(f'Batch of test_images has shape: {images.shape}')
    print(f'Batch of test_likes: {likes}')
    
    break  

resnet18_v2 = models.resnet18(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18_v2 = resnet18_v2.to(device)
num_ftrs = resnet18_v2.fc.in_features
# resnet18_v2.fc = nn.Linear(num_ftrs, 1)
# early_stopping = EarlyStopping(patience=7, verbose=True, path='resnet18_v2_checkpoint.pth')
resnet18_v2.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5), 
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  
)



criterion = nn.MSELoss()
optimizer = optim.Adam(resnet18_v2.parameters(), lr=0.001)


num_epochs = 10

resnet18_v2.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        
        outputs = resnet18_v2(inputs)
        loss = criterion(outputs.squeeze(), labels)

       
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

resnet18_v2.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18_v2(inputs).squeeze()

        # Calculate log of predicted and actual likes
        log_preds = torch.log(outputs)
        log_labels = torch.log(labels)

        # Calculate if log(predicted) is within ±20% of log(actual)
        lower_bound = log_labels * 0.8
        upper_bound = log_labels * 1.2

        within_range = (log_preds >= lower_bound) & (log_preds <= upper_bound)
        correct += within_range.sum().item()
        total += labels.size(0)

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')

# Final evaluation on the test set
resnet18_v2.eval()
test_relative_differences = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18_v2(inputs).squeeze()

        log_preds = torch.log(outputs)
        log_labels = torch.log(labels)

        relative_difference = torch.abs(log_preds - log_labels) / torch.abs(log_labels)
        test_relative_differences.append(relative_difference)

# Compute the average relative difference on the test set
average_relative_difference = torch.cat(test_relative_differences).mean().item()
print(f'Average Relative Difference on Test Set: {average_relative_difference*100:.4f}%')

torch.save(resnet18_v2.state_dict(), 'resnet18_v2_model.pth')

