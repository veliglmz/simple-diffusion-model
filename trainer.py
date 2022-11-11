import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import DenoisingModel

class Trainer:
    def __init__(self, device, n_epochs, batch_size, lr, T, beta0, betaT) -> None:
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size 
        self.lr = lr
        self.T = T
        self.beta0 = beta0 
        self.betaT = betaT
        
        self.betas = torch.linspace(self.beta0, self.betaT, T, device=self.device)
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.log().cumsum(0).exp()
        self.sigmas = self.betas.sqrt()

        self.model = DenoisingModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.X_train_mean = None
        self.X_train_std = None
    

    def train(self, X_train):
        print("Training is starting...")
        self.X_train_mean, self.X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
        print(f"Train data size: {X_train.size()}")
        print(f"Train data mean: {self.X_train_mean.cpu().detach().numpy()}")
        print(f"Train data std: {self.X_train_std.cpu().detach().numpy()}")
        print("#" * 20)

        self.model.train()
        for e in range(1, self.n_epochs+1):
            idx = torch.randint(0, self.T, (self.batch_size, ), device=self.device)
            x0 = X_train[idx]
            x0 = (x0 - self.X_train_mean) / self.X_train_std

            t = torch.randint(0, self.T, (self.batch_size, 1), device=self.device)
            t_enc = t / (self.T - 1) - 0.5
            eps = torch.randn_like(x0)

            xt = torch.sqrt(self.alphas_bar[t]) * x0 + torch.sqrt(1 - self.alphas_bar[t]) * eps     
            xt_enc = torch.concat((xt, t_enc), axis=1)
            outputs = self.model(xt_enc)

            loss = (eps - outputs).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if e % 500 == 0:
                print(f"Epoch {e}: {loss.item()}")
        torch.save(self.model.state_dict(), "denoising_model.pth")
        print("Training is done.")
        print("#" * 20)


    def test(self, X_test):
        print("Testing is starting...")
        self.model.eval()
        with torch.no_grad():
            for t in tqdm(range(self.T-1, -1, -1)):
                t_enc = torch.full((X_test.size(0), 1), t, device=self.device)
                t_enc = t_enc / (self.T - 1) - 0.5
                Xt_enc = torch.concat((X_test, t_enc), axis=1)
                output = self.model(Xt_enc)

                z = torch.zeros_like(X_test) if t == 0 else torch.randn_like(X_test)

                X_test = 1/torch.sqrt(self.alphas[t]) \
                        * (X_test - (1-self.alphas[t]) / torch.sqrt(1-self.alphas_bar[t]) * output) \
                        + self.sigmas[t] * z
        X_test = X_test * self.X_train_std + self.X_train_mean
        print("Testing is done.")
        print("#" * 20)
        return X_test