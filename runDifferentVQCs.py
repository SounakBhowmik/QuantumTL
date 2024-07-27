#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:42:16 2024

@author: sounakbhowmik
"""

#%%
# Imports
import torch.nn as nn
import numpy as np
import os
import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import pennylane as qml
import pickle


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
#%%
# Define data paths
resizedImage_folder = '../Datasets/NEU_DET/ReIMAGES'
resizedNoDefectImage_folder = '../Datasets/NEU_DET/ReNO_DEFECT'
#q_feature_file_path = 'Q_Feature_Maps/ALL_Q_featureMap_4x32x32.pt'
#q_feature_labels_file_path = 'Q_Feature_Maps/ALL_Q_featureMap_4x32x32_labels.pt'

input_dim = (200, 200)

def standardize_data(X):
    #X_standardized = (X-np.min(X))/(np.max(X) - np.min(X))
    X_standardized = X.reshape(X.shape[0], X.shape[-2], X.shape[-1])
    X_standardized = (X_standardized-np.mean(X_standardized))/np.std(X_standardized)
    return X_standardized.reshape(X.shape[0], 1, X.shape[-2], X.shape[-1])
#%% Prepare dataset. Resize to fit the criterion appropriate for Quanvolution

# Anomalies
selected_anomalies = ["rolled-in", "patches", "inclusion", "scratches"]
labels_ano = []
X_ano = []
for fname in os.listdir(resizedImage_folder):
    for i in range(len(selected_anomalies)):
        if(fname.split('_')[0] == selected_anomalies[i]):
            labels_ano.append(i+1)
            break
    fpath = os.path.join(resizedImage_folder, fname)
    im = cv2.imread(fpath, 0) # gray scale read
    im = cv2.resize(im, input_dim, interpolation = cv2.INTER_LINEAR)
    im = im.reshape(1, im.shape[0], im.shape[1])
    X_ano.append(im)
X_ano = np.array(X_ano)
labels_ano = np.array(labels_ano)

# Normal
X_norm = []
labels_norm = np.zeros(len(os.listdir(resizedNoDefectImage_folder)))
for fname in os.listdir(resizedNoDefectImage_folder):
    fpath = os.path.join(resizedNoDefectImage_folder, fname)
    im = cv2.imread(fpath, 0) # gray scale read
    im = cv2.resize(im, input_dim, interpolation = cv2.INTER_LINEAR)
    im = im.reshape(1, im.shape[0], im.shape[1])
    X_norm.append(im)
X_norm = np.array(X_norm)

#%% Binary Classification
# Selecting equanll number of normal images
random_idx = np.random.randint(low=0, high=X_norm.shape[0], size = (X_ano.shape[0]))
X_norm = X_norm[random_idx]
labels_norm = labels_norm[random_idx]

X = np.concatenate((X_norm, X_ano), axis = 0)
Y_poly = np.concatenate((labels_norm, labels_ano), axis = 0)
Y_bin = Y_poly
Y_bin [Y_bin > 0] = 1

# One-hot encode Y_bin
Y_bin = np.eye(2)[Y_bin.astype(int)]

# Normalize
X = X/255.0
# Standardize
# X = standardize_data(X)


#%%K-fold cross validate Quanvolution2D
from Models import QConv2D #, QConv2D_AE


class ConvModel_1(nn.Module):
    def __init__(self):
        super(ConvModel_1, self).__init__()
        self.name = "ConvModel_comapring_quanv"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 1, kernel_size=8, stride = 2)                      # -1, 1, 97, 97
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)                        # -1, 1, 47, 47

        #self.q_conv = QConv2D(in_channels=1, kernel_size=2, n_layers=3, stride=2)                    # -1, 4, 23,23

        self.conv_1_5= nn.Conv2d(1,4, kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(4, 16, kernel_size=4, stride = 1)                    # -1, 16, 20, 20
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)                        # -1, 16, 17, 17

        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride = 1)                    # -1, 32, 14, 14
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)                        # -1, 32, 13, 13

        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride = 2)                   # -1, 64, 5, 5

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)  # Output 2 value for binary classification
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,2)

    def forward(self, x):
        x = self.maxpool1(torch.tanh(self.conv1(x)))*torch.pi/2
        x = torch.relu(self.conv_1_5(x))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.maxpool3(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.relu(self.fc3(x))

        x = torch.log_softmax(self.fc4(x), dim=1)
        return x


# Building the model
class Hybrid_QuanvModel_1(nn.Module):
    def __init__(self):
        super(Hybrid_QuanvModel_1, self).__init__()
        self.name = "Hybrid_QuanvModel_circuit_14"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 1, kernel_size=8, stride = 2)                      # -1, 1, 97, 97
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)                        # -1, 1, 47, 47

        self.q_conv = QConv2D(in_channels=1, kernel_size=2, n_layers=1, stride=2, ckt_id=14)                    # -1, 4, 23,23

        self.conv2 = nn.Conv2d(4, 16, kernel_size=4, stride = 1)                    # -1, 16, 20, 20
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)                        # -1, 16, 17, 17

        #self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride = 1)                    # -1, 32, 13, 13
        #self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)                        # -1, 32, 13, 13

        #self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride = 2)                   # -1, 64, 5, 5

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 17 * 17, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)            
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,2)               # Output 2 value for binary classification

    def forward(self, x):
        x = self.maxpool1(torch.tanh(self.conv1(x)))*torch.pi/2
        x = torch.relu(self.q_conv(x))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        # x = self.maxpool3(torch.relu(self.conv3(x)))
        # x = torch.relu(self.conv4(x))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.relu(self.fc3(x))

        x = torch.log_softmax(self.fc4(x), dim=1)
        return x


#%% Initialize the hyperparameters and other training specs


##########
# Assuming X and Y are your dataset and labels (one-hot encoded)
X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
Y = torch.tensor(Y_bin, dtype=torch.long, requires_grad=False)  # Ensure Y is long for CrossEntropyLoss

# Convert one-hot encoded Y to class indices
Y_class_indices = torch.argmax(Y, dim=1)

# Hyperparameters
batch_size = 32
num_epochs = 50
validation_split = 0.2

# Create DataLoader
dataset = TensorDataset(X, Y_class_indices)
val_size = int(len(dataset) * validation_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#%%
learning_rate = 0.001

# Initialize model, loss function, and optimizer
model = Hybrid_QuanvModel_1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Create the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Dictionary to record performance metrics
performance = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'precision': None,
    'recall': None,
    'f1_score': None
}

# Training and validation loop
for epoch in tqdm(range(num_epochs)):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    scheduler.step()

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_Y.size(0)
            correct += (predicted == batch_Y).sum().item()
    
    val_accuracy = correct / total
    performance['train_loss'].append(train_loss / len(train_loader))
    performance['val_loss'].append(val_loss / len(val_loader))
    performance['val_accuracy'].append(val_accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')

# Calculate precision, recall, and F1 score
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_Y in val_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_Y.numpy())

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

performance['precision'] = precision
performance['recall'] = recall
performance['f1_score'] = f1

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Save performance metrics to a file
with open('RESULTS/test_different_VQC_designs/ckt_14_performance_metrics.pkl', 'wb') as f:
    pickle.dump(performance, f)


