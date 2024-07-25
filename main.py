# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:23:27 2024

@author: sbhowmi2
"""
#%%
# Imports
import torch.nn as nn
import pennylane.numpy as np
import os
import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import pennylane as qml
#%%
# Define data paths
resizedImage_folder = '../Datasets/NEU_DET/ReIMAGES'
resizedNoDefectImage_folder = '../Datasets/NEU_DET/ReNO_DEFECT'
#q_feature_file_path = 'Q_Feature_Maps/ALL_Q_featureMap_4x32x32.pt'
#q_feature_labels_file_path = 'Q_Feature_Maps/ALL_Q_featureMap_4x32x32_labels.pt'

input_dim = (200, 200)
# Test accuracy function
def test_acc(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    preds = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images) #-1x2
            predictions = np.argmax(outputs.detach().cpu().numpy(), axis = 1)
            correct += np.sum(np.logical_not(np.logical_xor(predictions, labels.numpy())))
            total += len(predictions)
            preds = preds+predictions.tolist()
            true_labels = true_labels+labels.tolist()

    precision = precision_score(true_labels, preds, average='binary')
    recall = recall_score(true_labels, preds, average='binary')
    f1 = f1_score(true_labels, preds, average='binary')

    accuracy = correct / total
    return accuracy, precision, recall, f1

def test_acc_loss(model, test_loader, criterion, device):
    '''
    return accuracy, precison, recall, f1_score, average test-loss per epoch
    '''
    model.eval()
    correct, total = 0, 0
    total_loss = 0
    preds = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images) #-1x2
            total_loss += criterion(outputs, torch.tensor(np.eye(2)[labels.cpu().numpy().astype(int)]))
            predictions = np.argmax(outputs.detach().cpu().numpy(), axis = 1)
            correct += np.sum(np.logical_not(np.logical_xor(predictions, labels.numpy())))
            total += len(predictions)
            preds = preds  + predictions.tolist()
            true_labels = true_labels + labels.tolist()
    precision = precision_score(true_labels, preds, average='binary')
    recall = recall_score(true_labels, preds, average='binary')
    f1 = f1_score(true_labels, preds, average='binary')
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader.dataset)
    return accuracy, precision, recall, f1, avg_loss
# Function to save the results
def saveResults(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
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

# Normalize
X = X/255.0
# Standardize
#X = standardize_data(X)

#%% Testing Quanvolution

from Models import Hybrid_QuanvModel
from sklearn.model_selection import train_test_split

# Define features and labels
X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(Y_bin, dtype = torch.float32)

# Define required parameters and functions
criterion = torch.nn.CrossEntropyLoss()  # For classification tasks, adjust accordingly
num_epochs = 50
batch_size = 64

# Define KFold cross validator

results_dict = {}
models = {}

# Define a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Hybrid_QuanvModel()
model.to(device)
model_name = "HybridModel_Quanvolution_based"
models[model_name] = None
results_dict[model_name] = {"loss": [], "test_acc": [], "precision": 0, "recall": 0, "F1_score": 0}
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split the data into training and validation sets for this fold
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

# Convert labels to one-hot encoding
y_train = torch.tensor(np.eye(2)[y_train.to(torch.int)], dtype=torch.float32)
print(y_train.shape)

# Convert to PyTorch DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

#%%
from tqdm import tnrange
for epoch in tnrange(50):  # num_epochs is a variable you should define
    total_loss = 0
    avg_loss = 0
    for batch_idx, (feature, label) in enumerate(train_loader):
        feature, label = feature.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(feature)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader.dataset)
    results_dict[model_name]["loss"].append(avg_loss)
    test_accuracy = test_acc(model, val_loader, device)
    results_dict[model_name]["test_acc"].append(test_accuracy)
    print(f"Loss: {avg_loss}, test_accuray: {test_accuracy}")
    
#%%
# Training done
# Calculate the precision, recall and f1 score of this model
true_labels = y_val.numpy()
predictions = model(X_val.to(device))
predicted_labels = np.argmax(predictions.detach().cpu().numpy(), axis = 1)
results_dict[model_name]["precision"] = precision_score(true_labels, predicted_labels, average='macro')  # Use 'micro', 'macro', 'weighted', or 'samples' for multi-class
results_dict[model_name]["recall"] = recall_score(true_labels, predicted_labels, average='macro')
results_dict[model_name]["F1_score"] = f1_score(true_labels, predicted_labels, average='macro')
#torch.save(model, f"RESULTS/HybridModels_k_fold_cross_val/{model_name}.pt")
models[model_name] = model

#%%
# Save the results_dict
results_file_path = 'RESULTS/HybridModel_Quanvolution_k_fold_cross_val/hybridModel_Quanvolution_based_k-fold_cross_val_results_dictionary.pkl'
saveResults(results_dict, results_file_path)


#%% TL-based models
from Models import Q_linear, DressedQuantumNet, DressedClassicalNet

cm_1 = torch.load('classical models/ClassicalModel1_1683709579050736241.pt', map_location=torch.device('cpu'))
cm_2 = torch.load('classical models/ClassicalModel2_11926550140455977890.pt', map_location=torch.device('cpu'))
cm_3 = torch.load('classical models/ClassicalModel3_6849243949351946134.pt', map_location=torch.device('cpu'))
cm_4 = torch.load('classical models/ClassicalModel4_4305358886776407372.pt', map_location=torch.device('cpu'))


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
#%%
#--------------------------------------------------------------------------------------------------------

import pickle
def saveResults(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score



def k_fold_cross_val(X, y, model_number, num_epochs = 50, n_splits = 6):
    criterion = torch.nn.CrossEntropyLoss()  # For classification tasks, adjust accordingly
    num_epochs = 50
    
    kf = KFold(n_splits = n_splits, shuffle=True, random_state=42)
    results_dict = {}
    models = {}
    
    # Define a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}")
    
        # Instantiating the hybrid model
        #model = build_hybrid_model(classical_model_path, DressedQuantumNet, device)
        #model = build_hybrid_model(cm_1, 1, 16)
        model = DressedClassicalNet(X.shape[-1])
        model_name = f"ClassicalModel{model_number}_transferLearning_based_{fold+1}"
        models[model_name] = None
        results_dict[model_name] = {"train_loss": [], "test_acc": [], "test_loss":[], "precision": [], "recall": [], "F1_score": []}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
        # Split the data into training and validation sets for this fold
        X_train, X_val = X[train_idx].clone().detach(), X[val_idx].clone().detach()
        y_train, y_val = y[train_idx], torch.tensor(y[val_idx], dtype=torch.float32)
        y_train = torch.tensor(np.array(np.eye(2)[y_train.astype(int)]), dtype=torch.float32)
    
    
        # Convert to PyTorch DataLoader
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
    
        # Training loop for this fold
        for epoch in tqdm(range(num_epochs)):  # num_epochs is a variable you should define
            total_loss = 0
            avg_loss = 0
            for batch_idx, (feature, label) in enumerate(train_loader):
                feature, label = feature.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(feature)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader.dataset)
            results_dict[model_name]["train_loss"].append(avg_loss)
            test_accuracy, precision, recall, f1, test_loss = test_acc_loss(model, val_loader, criterion, device)
            results_dict[model_name]["test_acc"].append(test_accuracy)
            results_dict[model_name]["test_loss"].append(test_loss)
            results_dict[model_name]["precision"].append(precision)
            results_dict[model_name]["recall"].append(recall)
            results_dict[model_name]["F1_score"].append(f1)
            print(f"Epoch: {epoch} of {fold} fold, test_loss: {test_loss}, test_accuracy: {test_accuracy}")
            
        models[model_name] = model
    
    # Save the results_dict
    results_file_path = f"RESULTS/ClassicalModel_TL_based_k_fold_cross_val/classicalModel{model_number}_transfer_learning_k-fold_cross_val_results_dictionary.pkl"
    saveResults(results_dict, results_file_path)

#%%
#Model 1 (M-2 as per the paper)
hm_1 = cm_1
for param in hm_1.parameters():
    param.requires_grad = False

hm_1 = nn.Sequential(hm_1.conv1,
                     nn.ReLU(),
                     hm_1.maxpool1,
                     hm_1.conv2,
                     nn.ReLU(),
                     hm_1.maxpool2,
                     hm_1.conv3,
                     nn.ReLU(),
                     Flatten())
X1 = hm_1(torch.tensor(X, dtype=torch.float32))
y = Y_bin
k_fold_cross_val(X1, y, model_number=1, num_epochs = 50, n_splits = 6)
#%%
#----------------------------------------------------------------------------------------------------
# Model 2 (Model-3 according to the paper)

hm_2 = cm_2
for param in hm_2.parameters():
    param.requires_grad = False

hm_2 = nn.Sequential(hm_2.conv1,
                     nn.ReLU(),
                     hm_2.maxpool1,
                     hm_2.conv2,
                     nn.ReLU(),
                     hm_2.maxpool2,
                     hm_2.conv3,
                     nn.ReLU(),
                     hm_2.maxpool3,
                     Flatten())

X2 = hm_2(torch.tensor(X, dtype=torch.float32))
y = Y_bin
k_fold_cross_val(X2, y, model_number=2, num_epochs = 50, n_splits = 6)

#%%
#----------------------------------------------------------------------------------------------------
# Model 3
hm_3 = cm_3
for param in hm_3.parameters():
    param.requires_grad = False

hm_3 = nn.Sequential(hm_3.conv1,
                     nn.ReLU(),
                     hm_3.maxpool1,
                     hm_3.conv2,
                     nn.ReLU(),
                     hm_3.maxpool2,
                     hm_3.conv3,
                     nn.ReLU(),
                     hm_3.maxpool3,
                     hm_3.conv4,
                     nn.ReLU(),
                     hm_3.maxpool4,
                     Flatten())

#%%
#-----------------------------------------------------------------------------------------------
# Model 4 (M-1 according to the paper)

hm_4 = cm_4
for param in hm_4.parameters():
    param.requires_grad = False

hm_4 = nn.Sequential(hm_4.conv1,
                     nn.ReLU(),
                     hm_4.maxpool1,
                     hm_4.conv2,
                     nn.ReLU(),
                     hm_4.maxpool2,
                     hm_4.conv3,
                     nn.ReLU(),
                     hm_4.maxpool3,
                     hm_4.conv4,
                     nn.ReLU(),
                     hm_4.maxpool4,
                     Flatten())

X4 = torch.concat((hm_4(torch.tensor(X[:1000],dtype=torch.float32)),
                  hm_4(torch.tensor(X[1000:2000],dtype=torch.float32)),
                  hm_4(torch.tensor(X[2000:(X.shape[0]+1)], dtype=torch.float32))), dim=0)
y = Y_bin

k_fold_cross_val(X4, y, model_number=4, num_epochs = 50, n_splits = 6)

#%%K-fold cross validate Quanvolution2D
from Models import QConv2D, QConv2D_AE
from sklearn.model_selection import KFold

results_dir = "HybridModel_Quanvolution_AE_k_fold_cross_val"

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
        self.name = "Hybrid_QuanvModel_AE"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 1, kernel_size=8, stride = 2)                      # -1, 1, 97, 97
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)                        # -1, 1, 47, 47

        self.q_conv = QConv2D_AE(in_channels=1, kernel_size=4, n_layers=3, stride=2)                    # -1, 4, 22,22

        self.conv2 = nn.Conv2d(4, 16, kernel_size=4, stride = 1)                    # -1, 16, 19, 19
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)                        # -1, 16, 16, 16

        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride = 1)                    # -1, 32, 13, 13
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
        x = torch.relu(self.q_conv(x))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.maxpool3(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.relu(self.fc3(x))

        x = torch.log_softmax(self.fc4(x), dim=1)
        return x

# 6-fold cross validation script
criterion = torch.nn.CrossEntropyLoss()  # For classification tasks, adjust accordingly
num_epochs = 50
batch_size = 64

# Convert your features and labels to PyTorch tensors
# X is X
y = Y_bin

kf = KFold(n_splits=6, shuffle=True, random_state=42)
results_dict = {}

# Define a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    model = Hybrid_QuanvModel_1()
    model_name = f"{model.name}_{fold+1}"
    results_dict[model_name] = {"test_acc": [], "test_loss":[], "precision": [], "recall": [], "F1_score": []}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Split the data into training and validation sets for this fold
    X_train, X_val = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[val_idx], dtype=torch.float32)
    y_train, y_val = y[train_idx], torch.tensor(y[val_idx], dtype=torch.float32)
    y_train = torch.tensor(np.array(np.eye(2)[y_train.astype(int)]), dtype=torch.float32)


    # Convert to PyTorch DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size = batch_size)

    # Training loop for this fold
    for epoch in tqdm(range(num_epochs)):  # num_epochs is a variable you should define
        
        for batch_idx, (feature, label) in enumerate(train_loader):
            feature, label = feature.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        test_accuracy, precision, recall, f1, test_loss = test_acc_loss(model, val_loader, criterion, device)
        results_dict[model_name]["test_acc"].append(test_accuracy)
        results_dict[model_name]["test_loss"].append(test_loss)
        results_dict[model_name]["precision"].append(precision)
        results_dict[model_name]["recall"].append(recall)
        results_dict[model_name]["F1_score"].append(f1)
        print(f"Epoch: {epoch} of {fold} fold, test_loss: {test_loss}, test_accuracy: {test_accuracy}")
    # Training done
    '''
    try:
        torch.save(model.state_dict(), f"RESULTS/{results_dir}/models_1_kfold/{model_name}.pt")
    except:
        print("Model was not saved.")
    '''
# Save the results_dict
results_file_path = f"RESULTS/{results_dir}/ConvModel1_for_comparison_k-fold_cross_val_results_dictionary.pkl"
saveResults(results_dict, results_file_path)











