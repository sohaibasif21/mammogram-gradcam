import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageEnhance
import csv
import os
import time  # for timing predictions

# === Streamlit Interface ===
st.set_page_config(page_title="Mammogram Classifier + Grad-CAM Report", page_icon="🩺", layout="wide")

# === Custom Styling ===
st.markdown("""
    <style>
    body {
        background-color: #f9fbff;
        font-family: 'Roboto', sans-serif;
    }
    .title {
        font-size: 36px;
        color: #003366;
        font-weight: 700;
        text-align: center;
    }
    .section-title {
        font-size: 20px;
        color: #005bac;
        font-weight: 600;
        margin-top: 20px;
    }
    .subtitle {
        font-size: 16px;
        color: #555;
        font-weight: 500;
    }
    .button, .download-btn {
        background-color: #009688;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .button:hover, .download-btn:hover {
        background-color: #00796b;
    }
    .footer-custom {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #e6f7f1;
        color: #333;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #ccc;
    }
    .reportview-container .main .block-container { padding-top: 2rem; }
    footer { visibility: hidden; }
    .sidebar .sidebar-content { padding-top: 2rem; }
    .file-uploader { padding: 10px; background-color: #f1f1f1; border-radius: 5px; }
    .sliders { margin-top: 15px; }
    .highlight { color: #d32f2f; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# === Load Model ===
@st.cache_resource
def load_models():
    classifier = load_model("MobileNetV2.h5", compile=False)
    return classifier

model = load_models()
class_labels = ["Density1+Benign", "Density1+Malignant", "Density2+Benign", "Density2+Malignant",
                "Density3+Benign", "Density3+Malignant", "Density4+Benign", "Density4+Malignant"]

# === Utilities ===
def preprocess_image(img, contrast_factor=1.0, zoom_factor=1.0, crop_area=None):
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    
    width, height = img.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    
    if crop_area:
        img = img.crop(crop_area)
    
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_class(img_array):
    preds = model.predict(img_array)[0]
    class_index = np.argmax(preds)
    return class_labels[class_index], preds[class_index], preds

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="multiply_1", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Getting the last convolution layer output
    last_conv_layer_output = last_conv_layer_output[0]
    
    # Calculating the heatmap
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# New Grad-CAM overlay function (overlay_heatmap_v2)
def overlay_heatmap_v2(original, heatmap, save_img_path="superimposed_img.jpg"):
    img = np.array(original.resize((224, 224)))
    # Resize heatmap to match the image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Normalize and apply a color map to the heatmap
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on the image
    superimposed_img = heatmap_colored * 0.4 + img

    # Save the superimposed image
    cv2.imwrite(save_img_path, superimposed_img)
    
    return superimposed_img

def log_to_csv(label, confidence, probs):
    path = "predictions_log.csv"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp, label, round(confidence, 4)] + [round(p, 4) for p in probs]
    header = ["Timestamp", "Label", "Confidence"] + [f"Class_{i}_Prob" for i in range(len(probs))]
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# === Title and Branding ===
st.markdown('<h1 class="title">🧠 Mammogram Breast Density Classifier + Grad-CAM Report</h1>', unsafe_allow_html=True)

# Hospital Branding
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown('<h3 class="subtitle">🏥 Taizhou Cancer Hospital</h3>', unsafe_allow_html=True)

# === Sidebar Instructions ===
st.sidebar.title("📋 Instructions")
st.sidebar.markdown("""
1. Upload a **mammogram image**.
2. The model will predict **density + malignancy**.
3. You'll see a **Grad-CAM** visualization and class probabilities.
4. Use the **interactive options** to adjust contrast, zoom, and crop.
""")

# === Sidebar Interactive Controls ===
contrast_factor = st.sidebar.slider("Adjust Contrast", 0.5, 2.0, 1.0)
zoom_factor = st.sidebar.slider("Adjust Zoom", 1.0, 2.0, 1.0)
crop_area = st.sidebar.text_input("Crop Area (left, upper, right, lower)", "0,0,224,224")
crop_area = tuple(map(int, crop_area.split(','))) if crop_area else None

# === File Uploader ===
uploaded_file = st.file_uploader("📤 Upload Mammogram Image", type=["png", "jpg", "jpeg"])

# === Tabs Layout ===
if uploaded_file:
    tabs = st.tabs(["📸 Uploaded Image", "📈 Results", "🔥 Grad-CAM", ⬇️ Download"])
    image_pil = PILImage.open(uploaded_file).convert("RGB")
    
    with tabs[0]:
        st.image(image_pil, caption="Uploaded Mammogram", use_column_width=True)
    
    with st.spinner("🔍 Predicting... Please wait..."):
        start_time = time.time()
        img_array = preprocess_image(image_pil, contrast_factor, zoom_factor)
        label, confidence, probs = predict_class(img_array)
        heatmap = make_gradcam_heatmap(img_array, model)
        grad_img = overlay_heatmap_v2(image_pil, heatmap, "gradcam.png")
        log_to_csv(label, confidence, probs)
        prediction_time = time.time() - start_time

    with tabs[1]:
        st.success(f"✅ **Prediction:** {label} ({confidence*100:.2f}%)")
        st.info(f"⏱️ **Prediction Time:** {prediction_time:.2f} seconds")
        st.markdown("#### 🔢 Class Probabilities:")
        for i, cls in enumerate(class_labels):
            st.write(f"• {cls}: `{probs[i]:.4f}`")
    
    with tabs[2]:
        st.markdown("#### 📍 Heatmap (Grad-CAM)")

        # Display Grad-CAM Overlay alongside Original Image
        grad_img_pil = PILImage.open("gradcam.png")

        # Create the plot to compare original and Grad-CAM image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Original Image
        ax[0].imshow(image_pil)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        # Grad-CAM Overlay Image
        ax[1].imshow(grad_img_pil)
        ax[1].set_title("Grad-CAM Overlay")
        ax[1].axis('off')

        st.pyplot(fig)

    with tabs[3]:
        st.markdown("#### 📥 Download Grad-CAM Overlay")
        
        # Providing the download button for the Grad-CAM image
        with open("gradcam.png", "rb") as file:
            st.download_button("📥 Download Grad-CAM Overlay", file, "gradcam.png", mime="image/png")

# === Footer ===
st.markdown("""
<div class="footer-custom">
    🔗 Developed by Sohaib Asif | 📧 Contact: songhb@zjcc.org.cn | 📍 Taizhou Cancer Hospital
</div>
""", unsafe_allow_html=True)
