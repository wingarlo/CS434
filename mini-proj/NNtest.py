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
import loadingdata as loadD

learnRate = 0.1
weight_dec = 0
drop_out = 0.2
moment = 0.5

if len(sys.argv) == 5:
	learnRate = float(sys.argv[1])
	weight_dec = float(sys.argv[2])
	drop_out = float(sys.argv[3])
	moment = float(sys.argv[4])
	
#LOAD DATA
testFiles = ["./data/general_test_instances.csv","./data/subject2_instances.csv","./data/subject7_instances.csv"]

x,y,id = loadD.loadCSV("./data/Subject_1.csv","./data/list_1.csv")
trainingFeaturesArray, trainingTargetsArray = loadD.concatHalfHour(x,y,id)
OriginalFeatureCount = len(trainingFeaturesArray[0])
trainingFeatures = torch.tensor(trainingFeaturesArray, dtype=torch.float)
trainingTargets = torch.tensor(trainingTargetsArray, dtype=torch.float)
batchSize = len(trainingFeaturesArray)
train = torch.utils.data.TensorDataset(trainingFeatures, trainingTargets)
train_loader = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=True)

validationFeaturesArray, validationTargetsArray = loadD.concatHalfHour(x,y,id)
validationFeatures = torch.tensor(validationFeaturesArray, dtype=torch.float)
validationTargets = torch.tensor(validationTargetsArray, dtype=torch.float)
batchSize = len(validationFeaturesArray)
validation = torch.utils.data.TensorDataset(validationFeatures, validationTargets)
validation_loader = torch.utils.data.DataLoader(validation, batch_size=batchSize, shuffle=True)

testingFeaturesArray = np.array(loadD.loadTestData(testFiles[0]))
testingFeatures = torch.tensor(testingFeaturesArray, dtype=torch.float)
testbatchSize = len(testingFeaturesArray)
print "testbatchSize " + repr(testbatchSize)
print 'test_size:', testingFeatures.size(), 'test_type:', testingFeatures.type()
print " "

for (X_train, y_train) in train_loader:
	print('X_train:', X_train.size(), 'type:', X_train.type())
	print('y_train:', y_train.size(), 'type:', y_train.type())
	break


#NETWORK DEFINITION
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(OriginalFeatureCount, 50)
		self.fc1_drop = nn.Dropout(drop_out)
		self.fc2 = nn.Linear(50, 50)
		self.fc2_drop = nn.Dropout(drop_out)
		self.fc3 = nn.Linear(50, 9)

	def forward(self, x):
		#x = x.view(-1, 32*32*3) #-1 means don't know how many rows to reshape to
		x = F.relu(self.fc1(x))
		x = self.fc1_drop(x)
		x = F.relu(self.fc2(x))
		x = self.fc2_drop(x)
		return F.log_softmax(self.fc3(x))

model = Net()

print('Learning Rate: {}'.format(learnRate))
optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=moment, weight_decay = weight_dec)

#print(model)

def train(epoch, log_interval=100):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		target = target.type(torch.LongTensor)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		#print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0]))

def validate(loss_vector, accuracy_vector):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		data, target = Variable(data, volatile=True), Variable(target)
		data = Variable(data.view(-1, OriginalFeatureCount))
		output = model(data)
		target = target.type(torch.LongTensor)
		val_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100.0 * float(correct) / float(len(validation_loader.dataset))
	accuracy_vector.append(accuracy)
	
	#print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss, correct, len(validation_loader.dataset), accuracy))
	return (float(val_loss), accuracy)

def test(testtensor,localbatchsize):
	model.eval()
	val_loss, correct = 0, 0
	data = Variable(testtensor, volatile=True)
	data = Variable(data.view(-1, OriginalFeatureCount))
	output = model(data)
	print "output size: " + repr(output.size())

	'''
	filename = 'MLPresult.csv'
	with open(filename,'wb') as mlpout:
		outputF = csv.writer(mlpout, delimiter=',')
		for q in range(len(results)):
			output.writerow([q]+[results[q][1]]+[results[q][0]])
	'''
	return (0)

def safeString(num,tag):
	if num < 1.0:
		temp = str(num)
		temp = tag+temp[2:]
	else:
		temp = str(num)
		temp = '.'.join(temp.split())
		temp = tag+temp
	return temp

#TRAINING
epochs = 10
results = []
lossv, accv = [], []
for epoch in range(1, epochs + 1):
	train(epoch)
	results.append(validate(lossv, accv))
	'''
	filename = 'MLPresult.csv'
	with open(filename,'wb') as mlpout:
		output = csv.writer(mlpout, delimiter=',')
		output.writerow(['Epoch']+['Accuracy']+['Average Loss'])
		for q in range(len(results)):
			output.writerow([q]+[results[q][1]]+[results[q][0]])
	'''

test(testingFeatures, testbatchSize)