import sys
sys.path.append('./pretrained-model/')
sys.path.append('./pretrained-model-data-augmentation/')
sys.path.append('./models/')

#from COVID_model_pretrained import *
from COVID_model_pretrained_data_augmentation import *
from sklearn.preprocessing import StandardScaler
import tqdm
import copy
import logging
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


def prepare_data (path_test):
	f = open(path_test, "r")
	real_test_files = []
	for x in f:
		[filename, label] = x.split(",")
		real_test_files.append(filename)

	f.close()

	
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                             	 std=[0.229, 0.224, 0.225])

	# NOT really positive and negative files (just for compatibility)! 
	real_test_dataset = MYDataset(path="../data/processed/test-whole-sampled", 
							      positive_files=real_test_files, 
							      negative_files=[], 
							      is_train=False,
							      transforms=normalize)

	dataloaders = {}
	dataloaders['real_test'] = DataLoader(real_test_dataset, batch_size=16, shuffle=False)
	return dataloaders, real_test_files


def test(model, dataloaders, args):
	model.eval()
	real_test_predictions = []
	for mode, dataloader in dataloaders.items():
		for (x, labels) in dataloader:
		    x = x.to(args['device'])
		    label = labels.to(args['device'])
		    pred = model(x)
		    if(mode == 'real_test'):
		        real_test_predictions.extend(list(pred.flatten().data.cpu().numpy()))    

	real_test_predictions = np.array(real_test_predictions)
	return real_test_predictions



if __name__ == "__main__":
	model_type = "data_augmentation"
	dataloaders, real_test_files = prepare_data("../data/test.txt")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Network()
	model.load_state_dict(torch.load("models/pretrained_model_data_augmentation.pt"))
	model = model.to(device)
	arguments = {'device':device}

	real_test_predictions = test(model, dataloaders, arguments)

	assert(len(real_test_files) == len(real_test_predictions))
	f = open("../data/test_predictions_" + model_type + ".txt", "w+")
	for file, prediction in zip(real_test_files, real_test_predictions):
		f.write(file + ", " + str(prediction) + "\n")

	f.close()