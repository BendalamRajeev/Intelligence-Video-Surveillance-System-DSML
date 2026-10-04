import streamlit as st
import torch
import cv2
import tempfile
from torchvision import transforms
from models.three_d_cnn_model import R3DViolenceDetector
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from ultralytics import YOLO

# Load model
# Load YOLO model for person detection
yolo_model = YOLO("yolov5s.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R3DViolenceDetector(pretrained=False).to(device)
model.load_state_dict(torch.load("r3d_model.pth", map_location=device))
model.eval()

# Grad-CAM hook variables
target_activations = None
gradients = None

def save_activation(module, input, output):
    global target_activations
    target_activations = output.detach()

def save_gradient(module, grad_input, grad_output):
    global gradients
    if grad_output[0] is not None:
        gradients = grad_output[0].detach()

model.r3d.layer4.register_forward_hook(save_activation)
model.r3d.layer4.register_backward_hook(save_gradient)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

clip_len = 16
stride = 8

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    return transform(img)

def compute_gradcam(clip_tensor):
    global target_activations, gradients
    target_activations = None
    gradients = None
    clip_tensor.requires_grad_(True)
    output = model(clip_tensor)
    model.zero_grad()
    output.backward(retain_graph=True)
    if gradients is None or target_activations is None:
        return np.zeros((7, 7), dtype=np.uint8)
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3, 4])
    for i in range(target_activations.shape[1]):
        target_activations[:, i] *= pooled_grads[i]
    heatmap = torch.mean(target_activations, dim=[1, 2]).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

lock_duration = 64  # Lock for 32 frames (adjustable)
locked_boxes = []  # Global list to store locked boxes

def overlay_heatmap_on_frame(frame, heatmap=None):
    global locked_boxes
    frame_height, frame_width = frame.shape[:2]
    overlayed = frame.copy()

    if heatmap is not None:
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (frame_width, frame_height))
        heatmap_mask = (heatmap_resized > 0.6).astype(np.uint8) * 255
        colormap = plt.cm.jet
        heatmap_colored = colormap(heatmap_resized)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        overlayed = cv2.addWeighted(overlayed, 0.6, heatmap_colored, 0.4, 0)
    else:
        heatmap_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    results = yolo_model.predict(frame, verbose=False)[0]
    boxes = results.boxes
    new_locked_boxes = []

    # Ensure we only draw for class 'person' (id=0)
    person_boxes = [box for box in boxes if int(box.cls) == 0]

    for box in person_boxes:
        cls_id = int(box.cls)
        if cls_id != 0:
            # Skip non-human objects, draw nothing
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if heatmap is not None:
            person_mask = heatmap_mask[y1:y2, x1:x2]
            overlap = np.mean(person_mask > 0)
            if overlap > 0.1:
                new_locked_boxes.append({"box": (x1, y1, x2, y2), "frames_left": lock_duration})
                cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(overlayed, "VIOLENT - Human", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue
        cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(overlayed, "Human", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    updated_locked_boxes = []
    for box_data in locked_boxes:
        box_data["frames_left"] -= 1
        if box_data["frames_left"] > 0:
            x1_new, y1_new, x2_new, y2_new = box_data["box"]
            found = False
            for box in person_boxes:
                cls_id = int(box.cls)
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                iou = (min(x2, x2_new) - max(x1, x1_new)) * (min(y2, y2_new) - max(y1, y1_new))
                if iou > 0:
                    x1_new, y1_new, x2_new, y2_new = x1, y1, x2, y2
                    found = True
                    break
            updated_locked_boxes.append({"box": (x1_new, y1_new, x2_new, y2_new), "frames_left": box_data["frames_left"]})
            cv2.rectangle(overlayed, (x1_new, y1_new), (x2_new, y2_new), (0, 0, 255), 2)
            cv2.putText(overlayed, "VIOLENT - Human", (x1_new, y1_new - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    locked_boxes = updated_locked_boxes + new_locked_boxes
    return overlayed

# The rest of your video processing and Streamlit UI goes below here...

st.title("🎩 R3D Violence Detection + Grad-CAM Video")
st.write("Upload a video. The model scans clips and highlights violent regions using Grad-CAM + YOLO people detection.")

uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    st.video(tfile.name)

    if st.button("Run Detection + Grad-CAM Video"):
        with st.spinner("Analyzing and generating Grad-CAM video..."):
            cap = cv2.VideoCapture(tfile.name)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_video_path = "gradcam_output.mp4"
            out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            buffer = []
            raw_buffer = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                buffer.append(preprocess_frame(frame))
                raw_buffer.append(frame)

                if len(buffer) == clip_len:
                    clip = torch.stack(buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)
                    clip.requires_grad_(True)
                    output = model(clip)
                    prob = torch.sigmoid(output).item()

                    if prob > 0.5:
                        heatmap = compute_gradcam(clip)
                        for f in raw_buffer[:stride]:
                            overlay = overlay_heatmap_on_frame(f.copy(), heatmap)
                            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                            out_video.write(bgr)
                    else:
                        for f in raw_buffer[:stride]:
                            overlay = overlay_heatmap_on_frame(f.copy(), None)
                            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                            out_video.write(bgr)

                    buffer = buffer[stride:]
                    raw_buffer = raw_buffer[stride:]

            for f in raw_buffer:
                overlay = overlay_heatmap_on_frame(f.copy(), None)
                bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                out_video.write(bgr)

            cap.release()
            out_video.release()

        if os.path.exists(output_video_path):
            st.success("✅ Done! Here's the annotated video:")
            st.video(output_video_path)

            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=file,
                    file_name="gradcam_output.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("❌ Failed to generate the output video.")
# Let me know if you want that full block corrected too.
