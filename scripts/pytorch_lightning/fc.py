import torch
import torch.nn.functional as F
from torchvision import datasets
import torchvision.datasets
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    # Esta funcion es para descargar los datos
    def prepare_data(self):

        torchvision.datasets.MNIST(
            root="dataset/",
            train=True,
            download=True
        )

        torchvision.datasets.MNIST(
            root="dataset/",
            train=False,
            download=True
        )

    def setup(self,stage):
        self.all_dataset = torchvision.datasets.MNIST(
            root="dataset/",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )

        self.train_ds, self.val_ds = random_split(self.all_dataset, [50000, 10000])
        self.test_ds = torchvision.datasets.MNIST(
            root="dataset/", 
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)

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
        accuracy = self.accuracy(scores.softmax(dim=-1), y)
        f1_scores = self.f1_score(scores.softmax(dim=-1), y)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_f1', f1_scores, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores.softmax(dim=-1), y)
        f1_scores = self.f1_score(scores.softmax(dim=-1), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_f1', f1_scores, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores.softmax(dim=-1), y)
        f1_scores = self.f1_score(scores.softmax(dim=-1), y)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        self.log('test_f1', f1_scores)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


dm = MnistDataModule(data_dir='dataset/', batch_size=16, num_workers=4)

trainer = pl.Trainer(accelerator='gpu', devices=[0], min_epochs=1, max_epochs=3, callbacks=[EarlyStopping(monitor='val_loss')])
model = NN(784, 10)
trainer.fit(model,dm) 
trainer.test(model,dm)
trainer.validate(model,dm)


