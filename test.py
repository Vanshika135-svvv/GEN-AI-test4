import torch
import torchvision.transforms as T
from PIL import Image
from models.generator import Generator

def test_single_image(image_path, model_path="gen.pth"):
    gen = Generator().to("cpu")
    gen.load_state_dict(torch.load(model_path, map_location="cpu"))
    gen.eval()

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    
    with torch.no_grad():
        output = gen(input_img)
        # Denormalize
        output = output * 0.5 + 0.5
        T.ToPILImage()(output.squeeze(0)).save("test_result.png")
    print("Output saved as test_result.png")

if __name__ == "__main__":
    test_single_image("your_test_image.jpg")