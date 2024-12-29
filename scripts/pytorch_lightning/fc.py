import torch
import torch.nn.functional as F
import torchvision.datasets 
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

class NN(pl.LightningModule):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

all_dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_ds, val_ds = random_split(all_dataset, [5000, 10000])
test_ds = torchvision.datasets.MNIST(
    root="dataset/", 
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(dataset = train_ds, batch_size = 16, shuffle=True)
test_loader = DataLoader(dataset = test_ds, batch_size = 16, shuffle = False)
val_loader = DataLoader(dataset = val_ds, batch_size = 16, shuffle = False)

trainer = pl.Trainer(accelerator='cpu', devices=[0], min_epochs=1, max_epochs=3, precision=16)
model = NN(784, 10)
trainer.fit(model,train_loader, val_loader)

