import torch
import torch.nn as nn
from torch.optim import Adam

from models import PolyNet

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
# testmnist = datasets.MNIST('~/',train=False,transform=trans)

train_loader = DataLoader(trainmnist,batch_size=64)
# test_loader = DataLoader(testmnist,batch_size=64)


model = PolyNet()

model.load_state_dict(torch.load('poly_net.pt'))
model.eval()


loss_fn = nn.CrossEntropyLoss()


total_test = []
total_accuracy = []

running_test_loss = []

running_test_accuracy = []

with torch.no_grad():
    for x,y in train_loader:
        x = x.view(-1,28*28)
        p = model(x)
        loss = loss_fn(p,y)
            
        running_test_loss.append(loss.item())

        running_test_accuracy.append((p.argmax(1) == y).sum()/y.shape[0])



test_loss = torch.mean(torch.Tensor(running_test_loss))
accuracy = torch.mean(torch.Tensor(running_test_accuracy))*100

print(f'Test Loss: {test_loss: .3f}')
print(f'Accuracty: {accuracy: .3f}')
