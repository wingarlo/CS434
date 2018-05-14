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
validation_loader = torch.utils.data.DataLoader(test, batch_size=10000, shuffle=True)



for (X_train, y_train) in train_loader:
	print('X_train:', X_train.size(), 'type:', X_train.type())
	print('y_train:', y_train.size(), 'type:', y_train.type())
	break






#NETWORK DEFINITION
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 100)
		self.fc1_drop = nn.Dropout(0.2)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		#x = x.view(-1, 32*32*3) #-1 means don't know how many rows to reshape to
		x = F.relu(self.fc1(x))
		x = self.fc1_drop(x)
		return F.log_softmax(self.fc2(x),dim=0)

model = Net()
if cuda:
	model.cuda()
	
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print(model)


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
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))

def validate(loss_vector, accuracy_vector):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, requires_grad=True), Variable(target)
		data = Variable(data.view(-1, 32*32*3))
		output = model(data)
		target = target.type(torch.LongTensor)
		val_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100. * correct / len(validation_loader.dataset)
	accuracy_vector.append(accuracy)
	
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))







#TRAINING
epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
	
	train(epoch)
	validate(lossv, accv)
