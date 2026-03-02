import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # This line now filters out folders and only keeps image files
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = T.Compose([
            T.Resize((256, 512)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files[index])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # Split wide image: Left is Input (sketch), Right is Target (photo)
        return img[:, :, :256], img[:, :, 256:]