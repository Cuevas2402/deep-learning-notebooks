import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torchvision import ToTensor
from model import Unet

from utils import (
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	check_accuracy,
#	save_predictions_as_imgs
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def train_fn(loader, model, optimizer, loss_fn):
	loop = tqdm(loader)
	for batch_idx, (data, targets) in enumerate(loop):
		data = data.to(device=DEVICE)
		targets = targets.float().unsqueeze(1).to(device=DEVICE)
		with torch.cuda.amp.autocast():
			predictions = model(data)
			loss = loss_fn(predictions, targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loop.set_postfix(loss=loss.item())



def main():
	train_transform = transforms.Compose([
		transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
		transforms.RandomRotation(degrees=35),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomVerticalFlip(p=0.1),
		transforms.Normalize(
			mean=[0.0,0.0,0.0],
			std=[1.0, 1.0, 1.0],
		),
		ToTensor()
	])

	val_transform = transforms.Compose([
		transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
		transforms.Normalize(
			mean=[0.0,0.0,0.0],
			std=[1.0, 1.0, 1.0],
		),
		ToTensor()
	])

	model = Unet(in_channels=3, out_channels=1).to(device=DEVICE)

	loss = nn.BCEWithLogitsLoss()

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_loader , val_loader = get_loaders(
		TRAIN_IMG_DIR, 
		TRAIN_MASK_DIR,
		VAL_IMG_DIR, 
		TRAIN_IMG_DIR, 
		BATCH_SIZE,
		train_transform,
		val_transform,
		NUM_WORKERS,
		PIN_MEMORY
	)

	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, loss)

		checkpoint = {
			'state_dict':model.state_dict(),
			'optimizer':optimizer.state_dict()
		}


		save_checkpoint(checkpoint)

		check_accuracy(val_loader, model, DEVICE)







if __name__ == "__main__":
	main()
