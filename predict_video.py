# predict_video.py

import streamlit as st
import torch
import cv2
import tempfile
from torchvision import transforms
from models.three_d_cnn_model import R3DViolenceDetector  # Make sure your filename matches!
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R3DViolenceDetector(pretrained=False).to(device)
model.load_state_dict(torch.load("r3d_model.pth", map_location=device))
model.eval()

# Frame transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Streamlit UI
st.title("🔍 Violence Detection (R3D)")
st.write("Upload a video. The upgraded R3D model scans the entire video using sliding windows to detect violence.")

uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi"])

clip_len = 16
stride = 8

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    return transform(img)

def predict_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()

    predictions = []

    for start in range(0, len(frames) - clip_len + 1, stride):
        clip = torch.stack(frames[start:start + clip_len])  # [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]

        with torch.no_grad():
            output = model(clip)
            prob = torch.sigmoid(output).item()
            predictions.append(prob)

    return predictions

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)

    if st.button("Run Violence Detection"):
        with st.spinner("Analyzing video with R3D model..."):
            predictions = predict_video(tfile.name)

        avg_prob = np.mean(predictions)
        max_prob = np.max(predictions)

        st.write(f"**Average probability:** `{avg_prob:.2f}`")
        st.write(f"**Max probability:** `{max_prob:.2f}`")

        if avg_prob > 0.65 and max_prob > 0.75:
            st.error(f"⚠️ Violence Detected! (avg: {avg_prob:.2f}, max: {max_prob:.2f})")
        else:
            st.success("✅ No violence detected in the scanned video.")

        st.line_chart(predictions)
        st.caption("Confidence scores for each sliding window clip.")
