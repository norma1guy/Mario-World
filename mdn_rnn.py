import torch
import torch.nn as nn

class MDN(torch.nn.Module) :

    def __init__(self,dim,n_hidden,gaussians):
        super.__init__()
        self.n_hidden = n_hidden
        self.gaussians = gaussians
        self.dim = dim

        self.z = nn.Sequential(
            nn.Linear(self.dim,self.n_hidden),
            nn.Tanh()
        )

        self.pi = nn.Linear(self.n_hidden,self.gaussians)
        self.mu = nn.Linear(self.n_hidden,self.gaussians)
        self.sigma = nn.Linear(self.n_hidden,self.gaussians)

    def forward(self,x):
        z = self.z(x)
        pi = nn.functional.softmax(self.pi(z),-1)
        sigma = torch.exp(self.sigma(z))
        mu = self.mu(z)

        return pi,sigma,mu


