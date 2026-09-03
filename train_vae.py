
from vae import VAE
import torch,time,random,os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

class Vision :

    def __init__(self,xdim,ydim,path,data=None):

        self.width = xdim
        self.height = ydim
        self.dim = xdim * ydim
        self.path = path
        self.model = None
        self.data_loader = data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.data_loader == None :
                    print('No DataLoader has been provided so loading MNIST by default\n')
                    transform = transforms.ToTensor()
                    dataset = datasets.MNIST(
                        root="./data",
                        train=True,
                        download=True,
                        transform=transform
                    )
                    self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        try :
            self.model = torch.load(path,weights_only=False)
            print('Model loaded\n')
        except Exception as e: 
            print('Exception occured\n')
            print(f'{e}\n')
            print(f'Checkpoint is missing because no model is trained. First train a model with {self.train.__name__}\n')

        


    def _vae_loss(self,reconstruction, x, mu, logvar):

        reconstruction_loss = F.binary_cross_entropy(
            reconstruction,
            x,
            reduction="sum"
        )

        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return reconstruction_loss + kl_loss

    def train(self,epochs) :

        if self.model :
            print('Do you want to retrain the model?(y/n) ')
            choice = input()
            while choice not in ['y','n'] :
                os.system('clear')
                print(f'Please enter correct choice(y,n) ')
                choice = input()
            

        if choice == 'y' or self.model == None :

            model = VAE(self.dim).to(device=self.device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-3,
            )

            start = time.time()
            for epoch in range(epochs):

                model.train()

                for x, _ in self.data_loader:

                    # Flatten 
                    x = x.view(x.size(0), -1).to(self.device)

                    optimizer.zero_grad()

                    reconstruction, mu, logvar = model(x)

                    loss = self._vae_loss(
                        reconstruction,
                        x,
                        mu,
                        logvar
                    )
                    loss = loss / x.size(0)
                    loss = loss / self.dim
                    loss.backward()

                    optimizer.step()

                print(
                    f"Epoch {epoch + 1}, "
                    f"Loss: {loss.item():.2f}"
                )

            torch.save(model,self.path)
            self.model = model
            end = time.time()

            print(f'Time taken for training : {end - start:.2f}')


    def show_reconstruction(self):

        if self.model :

            self.model.eval()

            x, labels = next(iter(self.data_loader))
            original = x[random.randint(0,len(x))]
            x_flat = original.view(1, -1).to(self.device)
            with torch.no_grad():
                reconstruction, mu, logvar = self.model(x_flat)

            reconstruction = reconstruction.view(self.width, self.height)


            fig, axes = plt.subplots(1, 2)

            axes[0].imshow(original.squeeze(), cmap="gray")
            axes[0].set_title(f"Original ({labels[0].item()})")
            axes[0].axis("off")

            axes[1].imshow(reconstruction.cpu(), cmap="gray")
            axes[1].set_title("Reconstruction")
            axes[1].axis("off")

            plt.show()
        else :
            print('No model present please perform training first\n')


class MDN(torch.nn.Module) :

    def __init__(self,layers,gaussians):
        super.__init__()
        self.layers = layers
        self.gaussians = gaussians

    
           