# FORENSIC.AI // TASK-04

### **Description**
Neural Forensic Engine utilizing Pix2Pix Conditional Generative Adversarial Networks (cGANs) to translate non-visible thermal signatures into photorealistic RGB reconstructions. This module is designed for forensic investigators to identify heat gradients, hidden devices, or structural anomalies through an AI-powered "Heat-to-Color" mapping protocol.

---

## **🌐 Live Demo**

## **Deployment Link:** [Open vercel Link](https://gen-ai-test4.vercel.app/?_vercel_share=ulXTINR5GqbQXSHQ0hMNBDBz5alHPvJ7#about)

---

## **🚀 Key Features**

* **Neural Reconstruction:** Translates forensic thermal sketches into high-fidelity RGB renders.
* **ONNX Optimized:** Runs on an optimized CPU-inference engine to ensure zero-latency deployment on Vercel.
* **Forensic Export:** Integrated secure evidence download system to save neural outputs as PNG files.
* **Modern Aesthetic:** Professional Glassmorphism UI with reactive loading states and Safari compatibility.
* **Zero-Torch Backend:** Lightweight server environment using ONNX Runtime to bypass 500MB storage limits.

---

## **🛠️ Tech Stack**

* **Backend:** Python 3.13, Flask
* **Inference Engine:** ONNX Runtime (CPU Optimized)
* **Pre-processing:** NumPy, Pillow (PIL)
* **Frontend:** HTML5, CSS3 (Modern Glassmorphism), JavaScript (Fetch API)
* **Deployment:** Vercel

---

## **🔬 Forensic Context**

* **The Investigation:** This module maps non-visible thermal data into a visible RGB spectrum, allowing investigators to see temperature gradients as distinct color ranges.
* **Use Cases:** Ideal for locating missing persons in low-light environments or detecting hidden structural weaknesses.
* **Neural Mapping:** The AI predicts texture from heat intensity, providing a visual "Heat-to-Orange/Red" mapping for rapid evidence analysis.

---

## **📝 Task Details**

* **Project:** Image-to-Image Translation with Pix2Pix
* **Reference:** Task-04 (Prodigy InfoTech)
* **Objective:** Implement a GAN model that translates images from one domain to another (Thermal to RGB).

---

## **📂 File Structure**
```text
PRODIGY_TASK4/
├── app.py              # Optimized ONNX Backend
├── model.onnx          # Neural Weights
├── model.onnx.data     # Weight Data
├── templates/

│   └── index.html      # Forensic UI
├── static/
│   └── style.css       # Glassmorphism Styles
├── requirements.txt    # Optimized Dependencies
└── README.md         # Documentation

```
---