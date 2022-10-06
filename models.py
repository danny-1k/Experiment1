import torch
import torch.nn as nn


class ActNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def save_model(self):
        torch.save(self.state_dict(),'act_net.pt')


class PolyNet(nn.Module):
    def __init__(self, degree=3):
        super().__init__()

        self.degree = degree

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    
    def forward(self, x):
        out_ = self.fc1(x)

        for i in range(2,self.degree+1):
            out_ += self.fc1(x**i)

        out = self.fc2(out_)

        for i in range(2, self.degree+1):
            out += self.fc2(out_**i)

        out_ = out


        out = self.fc3(out_)

        for i in range(2, self.degree+1):
            out += self.fc3(out_**i)

        
        return out

    def save_model(self):
        torch.save(self.state_dict(),'poly_net.pt')
    

class NoActNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    
    def save_model(self):
        torch.save(self.state_dict(),'no_act_net.pt')


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 10)

    
    def forward(self, x):
        x = self.fc1(x)
        return x


    def save_model(self):
        torch.save(self.state_dict(),'linear_net.pt')