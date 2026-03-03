from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import base64
import os
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Point to your new lightweight model file
MODEL_PATH = "model.onnx"

# Initialize ONNX Runtime Session with CPU optimization
def load_onnx_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Set session options to manage threads without needing torch
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            return ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            return None
    return None

ort_session = load_onnx_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not ort_session:
        return jsonify({"error": "Model missing on server"}), 500
    
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        
        # 1. Manual Preprocessing (Replacing torchvision)
        img = img.resize((256, 256))
        img_array = np.array(img).astype(np.float32) / 255.0
        # Normalize to [-1, 1]
        img_array = (img_array - 0.5) / 0.5
        # Change format from (H, W, C) to (C, H, W) and add Batch dimension
        input_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, :]

        # 2. Run Fast Inference
        outputs = ort_session.run(None, {'input': input_tensor})
        output = outputs[0]
        
        # 3. Post-processing: Denormalize back to [0, 1]
        output = (output * 0.5 + 0.5).clip(0, 1)
        
        # Convert NumPy array back to Image
        output_img_arr = (output.squeeze(0).transpose(1, 2, 0) * 255).astype(np.uint8)
        output_pil = Image.fromarray(output_img_arr)

        # 4. Encode to Base64 for the web UI
        buffered = BytesIO()
        output_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({"image": img_str})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)