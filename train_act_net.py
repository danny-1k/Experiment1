import torch
import torch.nn as nn
from torch.optim import Adam

from models import ActNet

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

from utils import save_metrics

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

trainmnist = datasets.MNIST('~/',train=True,transform=trans)
testmnist = datasets.MNIST('~/',train=False,transform=trans)

train_loader = DataLoader(trainmnist,batch_size=64)
test_loader = DataLoader(testmnist,batch_size=64)

print('data ready...')

model = ActNet()

epochs = 30
lr = 3e-3

loss_fn = nn.CrossEntropyLoss()

optim = Adam(model.parameters(),lr=lr)
print('Started training!')


best_loss = float('inf')

total_train = []
total_test = []
total_accuracy = []

for e in tqdm(range(epochs)):

    running_train_loss = []
    running_test_loss = []

    running_test_accuracy = []

    for x,y in train_loader:
        x = x.view(-1,28*28)
        p = model(x)
        loss = loss_fn(p,y)
        running_train_loss.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        for x,y in test_loader:
            x = x.view(-1,28*28)
            p = model(x)
            loss = loss_fn(p,y)
            
            running_test_loss.append(loss.item())

            running_test_accuracy.append((p.argmax(1) == y).sum()/y.shape[0])



    train_loss = torch.mean(torch.Tensor(running_train_loss))
    test_loss = torch.mean(torch.Tensor(running_test_loss))
    accuracy = torch.mean(torch.Tensor(running_test_accuracy))*100

    total_train.append(train_loss.item())
    total_test.append(test_loss.item())
    total_accuracy.append(accuracy.item())

    save_metrics(total_train,total_test, total_accuracy, 'act')
     
    print(f'Epoch : {e+1} Train-Loss : {train_loss:.5f} Test-Loss : {test_loss:.5f}')
    
    if test_loss < best_loss:
        model.save_model()
        best_loss = test_loss

print('Done training!')