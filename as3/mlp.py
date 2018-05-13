import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np


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


#LOAD DATA
batch_size = 10000*4 #10000 examples per batch file

train = torch.utils.data.TensorDataset(trainingFeatures, trainingTargets)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)


test = torch.utils.data.TensorDataset(testingFeatures, testingTargets)
validation_loader = torch.utils.data.DataLoader(test, batch_size=10000, shuffle=True)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break
'''
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray")
    plt.title('Class: '+str(y_train[i]))
	
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x))

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
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		val_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100. * correct / len(validation_loader.dataset)
	accuracy_vector.append(accuracy)
	
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))
		

epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)
	
batch_size = 32

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)	

pltplt..figurefigure((figsizefigsize=(5,3))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('validation loss')

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('validation accuracy');
'''