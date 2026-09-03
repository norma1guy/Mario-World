import torch
from mdn_rnn import MDN
import numpy as np


class Model :

    def __init__(self,dim,n_hidden,gaussians,xdata,ydata,epochs = 10000):

        self.dim = dim
        self.n_hidden = n_hidden
        self.gaussians = gaussians
        self.epochs = epochs
        self.xdata = torch.tensor(xdata,requires_grad=True)
        self.ydata = torch.tensor(ydata,requires_grad=True)


    def _gaussian_distribution(self,y,mu,sigma):

        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = - 0.5 * result * result 
        denom = 1 / np.sqrt(2 * np.pi)

        return (torch.exp(result) * torch.reciprocal(sigma)) * denom

    def _mdn_loss_fn(self,pi,sigma,mu,y) :

        result = self._gaussian_distribution(y,mu,sigma) * pi
        result = torch.sum(result, dim=1)
        result = -torch.log(result)
        return torch.mean(result)


    def train(self):

        network = MDN(self.dim,self.n_hidden,self.gaussians)
        optimizer = torch.optim.Adam(network.parameters())
        for epoch in range(self.epochs):
            pi,sigma,mu = network(self.xdata)
            loss = self._mdn_loss_fn(pi,sigma,mu,self.ydata)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0 :
                print(epoch, loss.data[0])
    
    



