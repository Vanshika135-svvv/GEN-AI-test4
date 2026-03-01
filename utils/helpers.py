import torch
from torchvision.utils import save_image
import os

def save_some_examples(gen, val_loader, epoch, folder="evaluation"):
    """
    Saves a side-by-side comparison of Input vs Generated vs Ground Truth.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    x, y = next(iter(val_loader))
    x, y = x.to(next(gen.parameters()).device), y.to(next(gen.parameters()).device)
    
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        # Denormalize from [-1, 1] to [0, 1]
        y_fake = y_fake * 0.5 + 0.5
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        
        # Stack images horizontally: [Input, Generated, Target]
        comparison = torch.cat([x, y_fake, y], dim=3)
        save_image(comparison, f"{folder}/result_epoch_{epoch}.png")
    gen.train()

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """Loads a saved model state."""
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    # If using an optimizer in training:
    # optimizer.load_state_dict(checkpoint["optimizer"])