import numpy as np
import pandas as pd
import nrrd
import nibabel as nib
import os
import pickle
import random

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cv2 import resize
from sklearn import metrics
 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import SGD, Adam

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = "test"
    info_file = "../../data/" + dataset + ".txt"
    path = "../../data/processed/" + dataset + "-whole"
    info_data_frame = pd.read_csv(info_file, header=None)

    # Feature extractor
    cutoff = 4
    embedding_size = 512
    histogram_size = 10
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    pretrained_model = torchvision.models.resnet34(pretrained=True)
    for param in pretrained_model.parameters():
        param.requires_grad = False

    pretrained_model.fc = nn.Identity()
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()


    files = info_data_frame.iloc[:, 0]
    labels = info_data_frame.iloc[:, 1]

    X = []
    y = []
    for file_name, label in zip(files, labels):
        CT_name = "/"+ file_name.strip('.nii.gz') + '.npy'
        file_path = path + CT_name
        img = np.load(file_path)
        color_data = img[img > 0].flatten()
        color_distribution = np.histogram(color_data, bins=10, range=(0, 1))[0] / color_data.size

        img = torch.tensor(img)
        img = img.repeat(1, 3, 1, 1)

        num_slices = img.shape[0]
        for slice_idx in range(cutoff, num_slices - cutoff):
            img[slice_idx, :, :, :] = normalize(img[slice_idx, :, :, :])

        img = img.to(device)
        extracted_features = pretrained_model(img).mean(dim=0)
        
        # Free GPU
        del img
        torch.cuda.empty_cache()

        features = np.append(extracted_features.cpu().detach().numpy(), color_distribution) 
        X.append(features)           
        y.append(label)

    
    X = np.vstack(X)
    y = np.array(y)

    data_frame = pd.DataFrame({"file":files,
                               "label":y
                             })



    print(data_frame.head())

    embedded_images = pd.DataFrame(X)
    embedded_images.columns = (["v" + str(i) for i in range(embedding_size)] + 
                               ["h" + str(i) for i in range(histogram_size)])

    data_frame = pd.concat([data_frame, embedded_images], axis=1)
    data_frame.to_csv("../../data/embedded-datasets/" + dataset + "_embedded_dataset.csv", index=False)