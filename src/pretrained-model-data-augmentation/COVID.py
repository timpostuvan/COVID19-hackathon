from COVID_model_pretrained_data_augmentation import *
from sklearn.preprocessing import StandardScaler
import tqdm
import copy
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def prepare_data (path_train):
	f = open(path_train, "r")
	positive_files = []
	negative_files = []
	#razdelimo v poz in negativne
	for x in f:
	    [filename, label] = x.split(",")
	    label = int(label)
	    if(label == 0):
	        negative_files.append(filename)
	    else :
	        positive_files.append(filename)

	f.close()

	random.shuffle(positive_files)
	plen = len(positive_files)
	positive_files_train = positive_files[0:int(plen*0.65)]
	positive_files_validation = positive_files[int(plen*0.65):int(plen*0.8)]
	positive_files_test = positive_files[int(plen*0.8):plen]

	random.shuffle(negative_files)
	nlen = len(negative_files)
	negative_files_train = negative_files[0:int(nlen*0.65)]
	negative_files_validation = negative_files[int(nlen*0.65):int(nlen*0.8)]
	negative_files_test = negative_files[int(nlen*0.8):nlen]

	data_augmentation = transforms.Compose([
							transforms.RandomApply(torch.nn.ModuleList([
								transforms.RandomRotation(degrees=15)]), 
							p=0.4),
							transforms.RandomApply(torch.nn.ModuleList([
								 transforms.RandomCrop(size=(200, 200)),
								 transforms.Pad(padding=12, fill=0, padding_mode='constant')]), 
							p=0.4),
					        transforms.RandomHorizontalFlip(p=0.4),
					        transforms.RandomVerticalFlip(p=0.4),
					        transforms.Normalize(mean=[0.485, 0.456, 0.406],
					                             std=[0.229, 0.224, 0.225]) 
					    ])

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                             	 std=[0.229, 0.224, 0.225])

	train_dataset = MYDataset(path="../../data/processed/train-whole-sampled", 
							  positive_files=positive_files_train, 
							  negative_files=negative_files_train, 
							  is_train=True,
							  transforms=data_augmentation)

	validation_dataset = MYDataset(path="../../data/processed/train-whole-sampled", 
	    						   positive_files=positive_files_validation, 
	    						   negative_files=negative_files_validation, 
	    						   is_train=False,
	    						   transforms=normalize)
	
	test_dataset = MYDataset(path="../../data/processed/train-whole-sampled", 
						     positive_files=positive_files_test, 
						     negative_files=negative_files_test, 
						     is_train=False,
						     transforms=normalize)


	dataloaders = {}
	dataloaders['train'] = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=5)
	dataloaders['val'] = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=5)
	dataloaders['test'] = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=5)
	return dataloaders


def train(model, dataloaders, optimizer, args, scheduler=None):
    # training loop
    val_max = -np.inf
    best_model = model
    train_scores = []
    val_scores = []
    test_scores = []
    for epoch in tqdm.tqdm(range(args['num_epochs'])):
        for iter_i, (x, color, labels) in enumerate(dataloaders['train']):
            x = x.to(args['device'])
            color = color.to(args['device'])
            labels = labels.to(args['device'])

            model.train()
            optimizer.zero_grad()
            pred = model(x, color)

            loss = model.loss(pred, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        scores, _ = test(model, dataloaders, args)
        train_scores.append(scores['train'])
        val_scores.append(scores['val'])
        test_scores.append(scores['test'])

        logging.info(log.format(epoch, scores['train'], scores['val'], scores['test']))
        if val_max < scores['val']:
            val_max = scores['val']
            best_model = copy.deepcopy(model)

    log = 'Best, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    scores, test_predictions = test(best_model, dataloaders, args)
    logging.info(log.format(scores['train'], scores['val'], scores['test']))
    return train_scores, val_scores, test_scores, test_predictions, best_model


def test(model, dataloaders, args):
	model.eval()
	scores = {}
	test_predictions = []
	for mode, dataloader in dataloaders.items():
		score = 0
		num_batches = 0
		all_batches_labels = []
		all_batches_pred = []
		for (x, color, labels) in dataloader:
		    x = x.to(args['device'])
		    color = color.to(args['device'])
		    label = labels.to(args['device'])

		    pred = model(x, color)
		    all_batches_pred.extend(list(pred.flatten().data.cpu().numpy()))
		    all_batches_labels.extend(list(labels.flatten().data.cpu().numpy()))

		    if(mode == 'test'):
		        test_predictions.extend(list(pred.flatten().data.cpu().numpy()))    

		  
		all_batches_labels = np.array(all_batches_labels)
		all_batches_pred = np.array(all_batches_pred)
		scores[mode] = metrics.roc_auc_score(all_batches_labels, all_batches_pred)

	test_prediction = np.array(test_predictions)
	return scores, test_predictions


if __name__ == "__main__":

	NUM_EXPERIMENTS = 3
	for i in range(NUM_EXPERIMENTS):
		
		# Set seeds for reproducibility
		random.seed(i)
		np.random.seed(i)
		torch.manual_seed(i)
	
		dataloaders = prepare_data("../../data/train.txt")

		# hyperparameters
		learning_rate = 0.005
		num_epochs = 20
	

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = Network()
		model = model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=0.01, cycle_momentum=False)
		arguments = {'device':device, 'num_epochs':num_epochs}

		
		train_scores, val_scores, test_scores, test_predictions, best_model = train(model, dataloaders, optimizer, arguments, scheduler)
		torch.save(best_model.state_dict(), f"../models/pretrained_model_data_augmentation_{i}.pt")

		epochs = list(range(len(train_scores)))
		plt.plot(epochs, train_scores, 'b-', label="train")
		plt.plot(epochs, val_scores, 'g-', label="validation")
		plt.plot(epochs, test_scores, 'r-', label="test")
		plt.legend()
		plt.savefig(f"graf_{i}")
		plt.clf()
	#	plt.show()