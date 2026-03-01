from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Use model.onnx instead of gen.pth for lightweight deployment
MODEL_PATH = "model.onnx"

# Initialize ONNX session
if os.path.exists(MODEL_PATH):
    # Vercel uses CPU, so we don't need to specify a device
    session = ort.InferenceSession(MODEL_PATH)
    print("ONNX Model loaded successfully!")
else:
    session = None
    print("WARNING: model.onnx not found. Run conversion script first.")

def preprocess(img):
    """Manually perform the T.Compose transforms using PIL and Numpy"""
    img = img.resize((256, 256))
    img_data = np.array(img).astype(np.float32) / 255.0
    # Normalize (0.5 mean, 0.5 std) -> (x - 0.5) / 0.5
    img_data = (img_data - 0.5) / 0.5
    # Change from HWC to CHW (Channels first)
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)

def postprocess(output_tensor):
    """Convert model output back to a displayable image"""
    # Denormalize: (x * 0.5) + 0.5
    output = (output_tensor.squeeze(0) * 0.5 + 0.5) * 255.0
    output = np.clip(output, 0, 255).astype(np.uint8)
    # Change from CHW back to HWC
    output = np.transpose(output, (1, 2, 0))
    return Image.fromarray(output)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    
    # 1. Preprocess
    input_data = preprocess(img)

    # 2. Run Inference with ONNX
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    raw_output = session.run([output_name], {input_name: input_data})[0]

    # 3. Postprocess
    res_img = postprocess(raw_output)

    # 4. Encode to Base64
    buffered = BytesIO()
    res_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({"image": img_str})

if __name__ == '__main__':
    app.run(debug=True)