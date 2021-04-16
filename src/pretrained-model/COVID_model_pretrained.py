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
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import SGD, Adam

class MYDataset(Dataset):
	# load the dataset
	def __init__(self, path, positive_files, negative_files, is_train, transforms=None):
		self.X = []
		self.y = []
		self.positive_files = positive_files
		self.negative_files = negative_files
		self.poslen = len(positive_files)
		self.neglen = len(negative_files)
		self.transforms = transforms
		self.num_slices = 20
		
		
		if(is_train):
			for i in range (0, self.neglen - self.poslen):
				index = random.randint(0, self.poslen - 1)
				self.positive_files.append(positive_files[index])

		for file in self.positive_files:
			current_slices = []
			for slice_idx in range(self.num_slices):
				slice_name = "/" + file.strip('.nii.gz') + "_slice" + str(slice_idx) + '.npy'
				slicepath = path + slice_name
				img = torch.tensor(np.load(slicepath))
				img = img.repeat(3, 1, 1)
				if(transforms):
					img = self.transforms(img)

				current_slices.append(img)
			self.X.append(torch.stack(current_slices, dim=0))			
			self.y.append(1.0)


		for file in self.negative_files:
			current_slices = []
			for slice_idx in range(self.num_slices):
				slice_name = "/" + file.strip('.nii.gz') + "_slice" + str(slice_idx) + '.npy'
				slicepath = path + slice_name
				img = torch.tensor(np.load(slicepath))
				img = img.repeat(3, 1, 1)
				if(transforms):
					img = self.transforms(img)

				current_slices.append(img)

			self.X.append(torch.stack(current_slices, dim=0))			
			self.y.append(0.0)

    	
	# number of rows in the dataset
	def __len__(self):
		return len(self.X)

	# get a row at an index
	def __getitem__(self, idx):
		return [self.X[idx], self.y[idx]]




class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.pretrained_model = torchvision.models.resnet34(pretrained=True)
		for param in self.pretrained_model.parameters():
			param.requires_grad = False

	#	print(self.pretrained_model)
		self.pretrained_model.fc = nn.Identity()

		self.linear = nn.Linear(in_features=512, out_features=1)
		self.activation = nn.Sigmoid()
		self.loss_fn = nn.BCELoss()

	def forward(self, x):
		batch_size = x.shape[0]
		predictions = []
		for i in range(batch_size):
			batch_xs = self.pretrained_model(x[i, :, :, :, :])
			batch_x = batch_xs.mean(dim=0)

			batch_x = self.linear(batch_x)
			batch_x = self.activation(batch_x)

			predictions.append(batch_x)

		preds = torch.stack(predictions, dim=0)
		return preds

	def loss(self, pred, label):
		return self.loss_fn(pred.flatten(), label.float())