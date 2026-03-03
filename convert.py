import torch
from models.generator import Generator

def convert_to_onnx(model_path="gen.pth", output_path="model.onnx"):
    device = "cpu"
    # 1. Initialize the model structure
    model = Generator().to(device)
    
    # 2. Load the trained weights from your .pth file
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Create a fake image (dummy input) to show ONNX the expected size
    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    # 4. Export it!
    torch.onnx.export(model, dummy_input, output_path, 
                      opset_version=11, 
                      input_names=['input'], 
                      output_names=['output'])
    print(f"✅ Successfully created {output_path}")

if __name__ == "__main__":
    convert_to_onnx()