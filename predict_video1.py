# predict_video.py

import streamlit as st
import torch
import cv2
import tempfile
from torchvision import transforms
from models.three_d_cnn_model import R3DViolenceDetector
from PIL import Image
import numpy as np

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
st.title("🔍 Violence Detection (R3D) — Robust Segments")
st.write(
    """
    Upload a video. The upgraded R3D model scans the entire video using sliding windows
    and detects short localized violence, not just average probability!
    """
)

uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi"])

clip_len = 16  # frames per clip
stride = 8     # overlap stride

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    return transform(img)

def predict_video(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buffer = []
    predictions = []

    progress_bar = st.progress(0)
    frame_idx = 0
    clip_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(preprocess_frame(frame))

        if len(buffer) == clip_len:
            clip = torch.stack(buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]
            with torch.no_grad():
                output = model(clip)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)

            buffer = buffer[stride:]  # slide window

            clip_idx += 1
            progress = int(clip_idx * stride / total_frames * 100)
            progress_bar.progress(min(progress, 100))

        frame_idx += 1

    cap.release()
    return predictions

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)

    if st.button("Run Violence Detection"):
        with st.spinner("Analyzing video with robust segment detection..."):
            predictions = predict_video(tfile.name)

        avg_prob = np.mean(predictions)
        max_prob = np.max(predictions)

        st.write(f"**Average probability:** `{avg_prob:.2f}`")
        st.write(f"**Max probability:** `{max_prob:.2f}`")

        # ✅ Smart robust segment logic
        clip_threshold = 0.5
        min_segments = max(1, int(0.05 * len(predictions)))  # at least 5% or at least 1

        violent_clips = [p for p in predictions if p > clip_threshold]

        st.write(f"Segments above threshold: {len(violent_clips)}/{len(predictions)}")

        if len(violent_clips) >= min_segments or max(predictions) > 0.6:
            st.error("⚠️ Violence Detected!")
        else:
            st.success("✅ No violence detected.")

        st.line_chart(predictions)
        st.caption("Confidence scores for each sliding window clip.")
