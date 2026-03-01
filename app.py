from flask import Flask, request, jsonify, render_template
import torch
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
import os
from models.generator import Generator

app = Flask(__name__)

# Check if model exists before starting
MODEL_PATH = "gen.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Generator().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
else:
    print("WARNING: gen.pth not found. Please run train.py first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet"}), 500
        
    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = output * 0.5 + 0.5 

    buffered = BytesIO()
    T.ToPILImage()(output.squeeze(0).cpu()).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({"image": img_str})

if __name__ == '__main__':
    app.run(debug=True)