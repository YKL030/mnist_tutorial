import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


class SimpleNet(nn.Module):
# TODO:define model
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.conv1 = nn.Sequential( nn.Conv2d(1,BATCH_SIZE,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(BATCH_SIZE,128,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = nn.Sequential( nn.Linear(14*14*128,1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(1024,10))
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1,14*14*128)
        x = self.dense(x)
        return x
    
model = SimpleNet()

# TODO:define loss function and optimiter
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# train and evaluate
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_correct = 0.0
    print("Epoch {}/{}".format(epoch,NUM_EPOCHS))
    print("-"*10)
    for images, labels in tqdm(train_loader):
    # TODO:forward + backward + optimize
        X_train = images
        Y_train = labels

        optimizer.zero_grad()
        #outputs:torch.FloatTensor = SimpleNet(X_train)
        outputs = model(X_train)
        _,pred = torch.max(outputs.data,1)
        loss = criterion(outputs,Y_train)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += torch.sum(pred == Y_train.data)
    testing_correct = 0.0
        
    for images, labels in tqdm(test_loader):
        X_test = images
        Y_test = labels
        outputs = model(X_test)
        _,pred = torch.max(outputs.data,1)
        testing_correct +=torch.sum(pred == Y_test.data)
    print("Loss is:{:.4f},Train Acurracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(running_loss/len(train_dataset),
                                                                                    100*running_correct/len(train_dataset),
                                                                                    100*testing_correct/len(test_dataset)))
    torch.save(model.state_dict(),"SimpleNet_parameter")    
        
    # evaluate
    # TODO:calculate the accuracy using traning and testing dataset

