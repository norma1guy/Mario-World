
import torch.nn as nn
import torch

class VAE(nn.Module) :

    def __init__(self,dim):

        super().__init__()
        

        #Encode
        self.fc = nn.Linear(dim,512)

        #Parameters mu and logvar
        self.mu = nn.Linear(512,128)
        self.logvar = nn.Linear(512,128)

        #Decode
        self.fc2 = nn.Linear(128,512)
        self.qx = nn.Linear(512,dim)


    def forward(self,x):

        h = nn.functional.relu(self.fc(x))
        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        h2 = nn.functional.relu(self.fc2(z))
        image = torch.sigmoid(self.qx(h2))
        return image,mu,logvar




