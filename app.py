# app.py

import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from train import FireCNN  # from your train.py

# Setup
st.set_page_config(page_title="🔥 Fire Detection Dashboard", layout="centered")
st.title("🔥 Fire Detection using CNN")
st.markdown("Upload an image or use webcam to detect fire.")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FireCNN().to(device)
model.load_state_dict(torch.load("fire_cnn.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Classes
classes = ["fire", "normal"]

# Function: Predict from image
def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)
    return classes[predicted.item()], conf.item()

# File upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(img)
    st.success(f"Prediction: **{label.upper()}** with confidence: **{confidence:.2f}**")

    if label == "fire":
        st.warning("🚨 FIRE DETECTED! Take immediate action.")
    else:
        st.info("✅ No fire detected.")

# Webcam detection
if st.checkbox("Use Webcam"):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        label, confidence = predict_image(image_pil)
        cv2.putText(frame, f"{label.upper()} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if label == "fire" else (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
