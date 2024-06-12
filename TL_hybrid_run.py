# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:52:30 2024

@author: sbhowmi2
"""
#%% Load and pre-process data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import albumentations.augmentations.functional as F
from Utilities.Extract_masks import create_filepaths
from PIL import Image, ImageOps
from tqdm import tqdm
import shutil
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import precision_score, recall_score, f1_score

image_folder = 'IMAGES'
annot_folder = 'ANNOTATIONS'
noDefect_folder = 'NO_DEFECT'
resizedImage_folder = 'ReIMAGES'
resizedNoDefectImage_folder = 'ReNO_DEFECT'
results_folder = "RESULTS"

x_dim, y_dim = (200, 200)

# Test accuracy function
def test_acc(model, test_loader, device):
    '''
    Takes in the Model, test-data-loader, and device (torch.Device())
    outputs accuracy, precision, recall and f1_score for the whole test data set
    '''
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

#%%Do not run this cell a second time unless you change the value of (x_dim, y_dim)

df = create_filepaths(annot_folder)
selected_anomalies = ["rolled-in_scale", "patches", "inclusion", "scratches"]
def test_acc(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images) #-1x2
            predictions = np.argmax(outputs.detach().cpu().numpy(), axis = 1)
            correct += np.sum(np.logical_not(np.logical_xor(predictions, labels.numpy())))
            total += len(predictions)

    accuracy = correct / total
    return accuracy

def load_preprocess_defective_images(df, image_folder, selected_anomalies, resizedImage_folder):
    subset = []
    shutil.rmtree(resizedImage_folder)
    os.mkdir(resizedImage_folder)
    for label in selected_anomalies:
        ls = df[df["Number_of_Defects"] == 1][df[label] == 1]["Name"].to_list()
        subset+=ls
    for fname in tqdm(subset):
        fpath = os.path.join(image_folder, (fname+".jpg"))
        image = Image.open(fpath)
        image = image.resize((x_dim, y_dim))
        image = ImageOps.grayscale(image)
        image.save(os.path.join(resizedImage_folder, (fname+".jpg")))

load_preprocess_defective_images(df, image_folder, selected_anomalies, resizedImage_folder)










