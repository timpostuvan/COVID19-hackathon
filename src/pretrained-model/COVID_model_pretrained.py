import numpy as np
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
import torch.optim as optim
from torch.optim import SGD, Adam

class MYDataset(Dataset):
	# load the dataset
	def __init__(self, path, positive_files, negative_files, is_train, transforms=None):
		self.X = []
		self.color = []
		self.y = []
		self.positive_files = positive_files
		self.negative_files = negative_files
		self.poslen = len(positive_files)
		self.neglen = len(negative_files)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')	
		self.transforms = transforms
		self.num_slices = 20

		# Pretrained model for feature extraction
		pretrained_model = torchvision.models.resnet34(pretrained=True)
		for param in pretrained_model.parameters():
			param.requires_grad = False

		pretrained_model.fc = nn.Identity()
		pretrained_model.to(self.device)
		

		for file in self.positive_files:
			current_slices = []
			color_data = []
			for slice_idx in range(self.num_slices):
				slice_name = "/" + file.strip('.nii.gz') + "_slice" + str(slice_idx) + '.npy'
				slicepath = path + slice_name
				img = torch.tensor(np.load(slicepath))
				color_data.append(img[img > 0].flatten())			

				img = img.repeat(3, 1, 1).view(1, 3, 224, 224)
				if(self.transforms):
					img = self.transforms(img)

				img = img.to(self.device)
				current_slices.append(pretrained_model(img).view(-1))

			color_data = np.concatenate(color_data, axis=0)
			color_distribution = np.histogram(color_data, bins=10, range=(0, 1))[0] / color_data.size

			self.X.append(torch.stack(current_slices, dim=0))	
			self.color.append(color_distribution)			
			self.y.append(1.0)


		for file in self.negative_files:
			current_slices = []
			color_data = []
			for slice_idx in range(self.num_slices):
				slice_name = "/" + file.strip('.nii.gz') + "_slice" + str(slice_idx) + '.npy'
				slicepath = path + slice_name
				img = torch.tensor(np.load(slicepath))
				color_data.append(img[img > 0].flatten())			

				img = img.repeat(3, 1, 1).view(1, 3, 224, 224)
				if(self.transforms):
					img = self.transforms(img)

				img = img.to(self.device)
				current_slices.append(pretrained_model(img).view(-1))

			color_data = np.concatenate(color_data, axis=0)
			color_distribution = np.histogram(color_data, bins=10, range=(0, 1))[0] / color_data.size

			self.X.append(torch.stack(current_slices, dim=0))
			self.color.append(color_distribution)			
			self.y.append(0.0)
    	
	# number of rows in the dataset
	def __len__(self):
		return len(self.X)

	# get a row at an index
	def __getitem__(self, idx):
		return [self.X[idx], self.color[idx], self.y[idx]]



class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()


class Network(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.NN = nn.Sequential(
				nn.BatchNorm1d(num_features=522),
				nn.Linear(in_features=522, out_features=1),
				nn.Sigmoid()
			)
		self.loss_fn = WeightedFocalLoss(alpha=0.18, gamma=1.5)

	def forward(self, x, color):
		batch_size = x.shape[0]
		predictions = []
		slices = []
		for i in range(batch_size):
			batch_xs = x[i, :, :]
			batch_x = batch_xs.mean(dim=0)

			batch_x = torch.cat([batch_x, color[i, :].float()], dim=0)
			slices.append(batch_x)

		x = torch.stack(slices, dim=0)
		preds = self.NN(x)
		return preds

	def loss(self, pred, label):
		return self.loss_fn(pred.flatten(), label.float())