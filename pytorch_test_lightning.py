# adapted from: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# This code should overfit to the test data since it is trained on it
# This code is meant to test GPU functionality

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import pytorch_lightning as pl

class LitNeuralNetwork(pl.LightningModule):
    def __init__(self, loss_fn, learning_rate):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
    @property
    def device(self):
        return next(self.parameters()).device 
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    #def backward(self, trainer, loss, optimizer, optimizer_idx):
    #    self.loss.
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        #if batch_idx % 100 == 0:
        #    loss, current = loss.cpu().item(), batch_idx * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}]")
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        return loss, correct
#    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64
print('data loaders')
train_dataloader = DataLoader(training_data, batch_size=64, num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=64, num_workers=16)
print(test_dataloader)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device='cpu'
print(f"Using {device} device")

print('themodel')
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
model = LitNeuralNetwork(loss_fn, learning_rate)
#model.configure_optimizer(learning_rate=lr)
print(model)
print(model.device)
epochs=5
trainer = pl.Trainer(accelerator="gpu",devices=[2],max_epochs=epochs)

start = time.time()
trainer.fit(model, train_dataloader, test_dataloader)

#for t in range(epochs):
#    print(f"Epoch {t+1}\n-------------------------------")
#    train_loop(train_dataloader, model, loss_fn, optimizer)
#    test_loop(test_dataloader, model, loss_fn)
print(f"{time.time()-start} elapsed")
print("Done!")

