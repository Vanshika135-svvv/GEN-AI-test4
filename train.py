import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import Pix2PixDataset
from utils.helpers import save_some_examples
from tqdm import tqdm

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 60
L1_LAMBDA = 100

def train_fn():
    # 1. Init Models & Optimizers
    disc = Discriminator().to(DEVICE)
    gen = Generator().to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # 2. Losses
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # 3. Data
    train_dataset = Pix2PixDataset(root_dir="data/train/train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Loop
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # --- Train Discriminator ---
            fake_y = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, fake_y.detach())
            loss_D = (BCE(D_real, torch.ones_like(D_real)) + BCE(D_fake, torch.zeros_like(D_fake))) / 2
            
            opt_disc.zero_grad()
            loss_D.backward()
            opt_disc.step()

            # --- Train Generator ---
            D_fake = disc(x, fake_y)
            loss_G = BCE(D_fake, torch.ones_like(D_fake)) + L1_LAMBDA * L1_LOSS(fake_y, y)
            
            opt_gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

        # Save model and preview every 5 epochs
        if epoch % 5 == 0:
            torch.save(gen.state_dict(), "gen.pth")
            save_some_examples(gen, train_loader, epoch, folder="evaluation")

if __name__ == "__main__":
    train_fn()