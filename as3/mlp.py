import sys
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

#dict["data"] contains 2D array of examples
#dict["labels"] contains labels of each example 0-9
#batches.meta: ["label_names"] contains corresponding name to the 0-9 labels
def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

learnRate = 0.1	
weight_dec = 0
drop_out = 0.2
moment = 0.5

if len(sys.argv) == 5:
	learnRate = float(sys.argv[1])
	weight_dec = float(sys.argv[2])
	drop_out = float(sys.argv[3])
	moment = float(sys.argv[4])

classes = unpickle("data/batches.meta")["label_names"]
batch1 = unpickle("data/data_batch_1")
batch2 = unpickle("data/data_batch_2")
batch3 = unpickle("data/data_batch_3")
batch4 = unpickle("data/data_batch_4")
testbatch = unpickle("data/test_batch")

trainingFeatures = np.concatenate((batch1["data"], batch2["data"], batch3["data"], batch4["data"]), axis=0)/255.0
trainingFeatures = torch.tensor(trainingFeatures, dtype=torch.float)
trainingTargets = torch.tensor(np.concatenate((batch1["labels"], batch2["labels"], batch3["labels"], batch4["labels"])), dtype=torch.float)
testingFeatures = testbatch["data"]
testingFeatures = torch.tensor(testingFeatures/255.0, dtype=torch.float)
testingTargets = torch.tensor(testbatch["labels"], dtype=torch.float)


cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

#LOAD DATA
batchSize = 10000 #10000 examples per batch file

train = torch.utils.data.TensorDataset(trainingFeatures, trainingTargets)
train_loader = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=True)

test = torch.utils.data.TensorDataset(testingFeatures, testingTargets)
validation_loader = torch.utils.data.DataLoader(test, batch_size=batchSize, shuffle=True)


for (X_train, y_train) in train_loader:
	print('X_train:', X_train.size(), 'type:', X_train.type())
	print('y_train:', y_train.size(), 'type:', y_train.type())
	break


#NETWORK DEFINITION
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 100)
		self.fc1_drop = nn.Dropout(drop_out)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		#x = x.view(-1, 32*32*3) #-1 means don't know how many rows to reshape to
		x = F.relu(self.fc1(x))
		x = self.fc1_drop(x)
		return F.log_softmax(self.fc2(x))
class Net2(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 50)
		self.fc1_drop = nn.Dropout(drop_out)
		self.fc2 = nn.Linear(50, 50)
		self.fc2_drop = nn.Dropout(drop_out)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x):
		#x = x.view(-1, 32*32*3) #-1 means don't know how many rows to reshape to
		x = F.relu(self.fc1(x))
		x = self.fc1_drop(x)
		x = F.relu(self.fc2(x))
		x = self.fc2_drop(x)
		return F.log_softmax(self.fc3(x))

model = Net()

optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=moment, weight_decay = weight_dec)

#print(model)

def train(epoch, log_interval=100):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		target = target.type(torch.LongTensor)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.data[0]))

def validate(loss_vector, accuracy_vector):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		data = Variable(data.view(-1, 32*32*3))
		output = model(data)
		target = target.type(torch.LongTensor)
		val_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100.0 * float(correct) / float(len(validation_loader.dataset))
	accuracy_vector.append(accuracy)
	
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))
	return accuracy
	
def buildNet(lr,wd,do,mo):
	learnRate = lr
	weight_decay = wd
	dropout = do
	momentum = mo
	model = Net()
	optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=moment, weight_decay = weight_dec)

	#TRAINING
	epochs = 10
	results = []
	lossv, accv = [], []
	for epoch in range(1, epochs + 1):
		train(epoch)
		results.append(validate(lossv, accv))
	return results

def buildNet2(lr,wd,do,mo):
	learnRate = lr
	weight_decay = wd
	dropout = do
	momentum = mo
	model = Net2()
	optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=moment, weight_decay = weight_dec)

	#TRAINING
	epochs = 10
	results = []
	lossv, accv = [], []
	for epoch in range(1, epochs + 1):
		train(epoch)
		results.append(validate(lossv, accv))
	return results

learnRate = 0.1	
weight_dec = 0
drop_out = 0.2
moment = 0.5
learn_ratearray = [0.9,0.1,0.01,0.001,0.0001]
weight_decayarray = [0., 0.001, 0.01, 0.1, 0.3, 0.7, 0.9, 1.0]
drop_outarray = [0.9, 0.5, 0.2, 0.1, 0.01, 0.0001, 0.000001]
momentumarray = [0., 0.25,0.5,0.75,0.9,0.0001,1.0,2.0,3.0]

LRResults = []
for a in learn_ratearray:
	LRResults.append([a]+buildNet(a,weight_dec,drop_out,moment))
filename = 'MLP_LR.csv'
with open(filename,'wb') as mlpout:
	output = csv.writer(mlpout, delimiter=',')
	for q in range(len(LRResults)):
		output.writerow([LRResults[q][0]]+[LRResults[q][1]]+[LRResults[q][2]]+[LRResults[q][3]]+[LRResults[q][4]]+[LRResults[q][5]]+[LRResults[q][6]]+[LRResults[q][7]]+[LRResults[q][8]]+[LRResults[q][9]]+[LRResults[q][10]])

WDResults = []
for b in weight_decayarray:
	WDResults.append([b]+buildNet(learnRate,b,drop_out,moment))
filename = 'MLP_WD.csv'
with open(filename,'wb') as mlpout:
	output = csv.writer(mlpout, delimiter=',')
	for q in range(len(WDResults)):
		output.writerow([WDResults[q][0]]+[WDResults[q][1]]+[WDResults[q][2]]+[WDResults[q][3]]+[WDResults[q][4]]+[WDResults[q][5]]+[WDResults[q][6]]+[WDResults[q][7]]+[WDResults[q][8]]+[WDResults[q][9]]+[WDResults[q][10]])

DOResults = []
for c in drop_outarray:
	DOResults.append([c]+buildNet(learnRate,weight_dec,c,moment))
filename = 'MLP_DO.csv'
with open(filename,'wb') as mlpout:
	output = csv.writer(mlpout, delimiter=',')
	for q in range(len(DOResults)):
		output.writerow([DOResults[q][0]]+[DOResults[q][1]]+[DOResults[q][2]]+[DOResults[q][3]]+[DOResults[q][4]]+[DOResults[q][5]]+[DOResults[q][6]]+[DOResults[q][7]]+[DOResults[q][8]]+[DOResults[q][9]]+[DOResults[q][10]])

MOResults = []
for d in drop_outarray:
	MOResults.append([d]+buildNet(learnRate,weight_dec,drop_out,d))
filename = 'MLP_MO.csv'
with open(filename,'wb') as mlpout:
	output = csv.writer(mlpout, delimiter=',')
	for q in range(len(MOResults)):
		output.writerow([MOResults[q][0]]+[MOResults[q][1]]+[MOResults[q][2]]+[MOResults[q][3]]+[MOResults[q][4]]+[MOResults[q][5]]+[MOResults[q][6]]+[MOResults[q][7]]+[MOResults[q][8]]+[MOResults[q][9]]+[MOResults[q][10]])

P4Results = []
for a in learn_ratearray:
	P4Results.append([a]+buildNet2(a,weight_dec,drop_out,moment))
filename = 'MLP_P4.csv'
with open(filename,'wb') as mlpout:
	output = csv.writer(mlpout, delimiter=',')
	for q in range(len(P4Results)):
		output.writerow([P4Results[q][0]]+[P4Results[q][1]]+[P4Results[q][2]]+[P4Results[q][3]]+[P4Results[q][4]]+[P4Results[q][5]]+[P4Results[q][6]]+[P4Results[q][7]]+[P4Results[q][8]]+[P4Results[q][9]]+[P4Results[q][10]])

