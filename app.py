from flask import Flask, request, jsonify, render_template
import torch
import base64
import os
import _pickle
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
from models.generator import Generator

app = Flask(__name__)

# Config
MODEL_PATH = "gen.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Model
model = Generator().to(device)

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Attempting to load {MODEL_PATH} on {device}...")
            # Using weights_only=False because custom U-Net classes 
            # often require legacy loading in newer PyTorch versions
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            print("✅ Model loaded successfully!")
            return True
        except (EOFError, _pickle.UnpicklingError, RuntimeError) as e:
            print(f"❌ ERROR: {MODEL_PATH} is corrupted or incomplete.")
            print("Action: Delete 'gen.pth' and run train.py for at least 1 epoch.")
            return False
    else:
        print("⚠️ WARNING: gen.pth not found. Please run train.py first.")
        return False

# Initial load attempt
is_model_ready = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model file missing. Please train the model."}), 500
    
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        
        # Pix2Pix standard preprocessing
        transform = T.Compose([
            T.Resize((256, 256)), 
            T.ToTensor(), 
            T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            # Denormalize: from [-1, 1] back to [0, 1]
            output = output * 0.5 + 0.5 

        # Convert back to Base64 for the frontend
        buffered = BytesIO()
        T.ToPILImage()(output.squeeze(0).cpu()).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({"image": img_str})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # We use debug=False here sometimes because it can trigger 
    # double-loading of the model, which causes issues on some GPUs
    app.run(debug=True, port=5000)