import os
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rebuilddata import CatsVsDogs

REBUILD_DATA = False

if REBUILD_DATA:
    CatsVsDogs.make_training_data()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the GPU")
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  #input, ouput, kernel size 5X5
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)  
        x = torch.randn(50,50).view([-1,1,50,50])
        self.__to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self.__to_linear, 512)
        self.fc2 = nn.Linear(512,2)
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        print(x[0].shape)
        if self.__to_linear is None:
            self.__to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.__to_linear)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)
        return x
net = Net().to(device)
a = np.load("training_data.npy", allow_pickle=True)

optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()
X = torch.Tensor([i[0] for i in a]).view(-1, 50 ,50)
X = X/255.0
y = torch.Tensor([i[1] for i in a])
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]
EPOCHS = 5
BATCH_SIZE = 100



def train(net):
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm.trange(0, len(train_X), BATCH_SIZE):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return loss
    print(loss)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm.trange(len(test_X)):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1,1,50,50).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
            print(correct/total*100)
loss = train(net)
test(net)

torch.save({
    'epoch': EPOCHS,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, "model.pth")