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

# Load model
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
    print(f"DEBUG: Activation saved - shape: {target_activations.shape}")

def save_gradient(module, grad_input, grad_output):
    global gradients
    if grad_output[0] is not None:
        gradients = grad_output[0].detach()
        print(f"DEBUG: Gradient saved - shape: {gradients.shape}")
    else:
        print("DEBUG: No gradient received!")

# Register hooks
model.r3d.layer4.register_forward_hook(save_activation)
model.r3d.layer4.register_backward_hook(save_gradient)

# Frame transform
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
    
    print("DEBUG: Computing Grad-CAM...")
    
    # Reset gradients
    target_activations = None
    gradients = None
    
    # Forward pass
    clip_tensor.requires_grad_(True)
    output = model(clip_tensor)
    print(f"DEBUG: Model output: {output.item()}")
    
    # Backward pass
    model.zero_grad()
    output.backward(retain_graph=True)
    
    # Check if we got gradients
    if gradients is None:
        print("ERROR: No gradients captured!")
        return np.zeros((7, 7), dtype=np.uint8)
    
    if target_activations is None:
        print("ERROR: No activations captured!")
        return np.zeros((7, 7), dtype=np.uint8)
    
    print(f"DEBUG: Gradients shape: {gradients.shape}")
    print(f"DEBUG: Activations shape: {target_activations.shape}")
    
    # Compute gradients
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3, 4])  # [C]
    print(f"DEBUG: Pooled gradients shape: {pooled_grads.shape}")
    
    # Apply gradients to activations
    for i in range(target_activations.shape[1]):
        target_activations[:, i] *= pooled_grads[i]
    
    # Generate heatmap - handle temporal dimension properly
    # target_activations shape: [1, 512, T, H, W] where T is temporal dimension
    # We need to average over both channel (dim=1) and temporal (dim=2) dimensions
    heatmap = torch.mean(target_activations, dim=[1, 2]).squeeze()  # [H, W]
    heatmap = heatmap.cpu().numpy()
    
    print(f"DEBUG: Initial heatmap shape: {heatmap.shape}")
    print(f"DEBUG: Heatmap min: {heatmap.min()}, max: {heatmap.max()}")
    
    # Process heatmap
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Ensure it's 2D
    if heatmap.ndim == 0:
        heatmap = np.zeros((7, 7))
    elif heatmap.ndim == 1:
        side = int(np.sqrt(len(heatmap)))
        if side * side == len(heatmap):
            heatmap = heatmap.reshape(side, side)
        else:
            heatmap = np.zeros((7, 7))
    elif heatmap.ndim > 2:
        # If still more than 2D, take the mean across extra dimensions
        while heatmap.ndim > 2:
            heatmap = np.mean(heatmap, axis=0)
    
    print(f"DEBUG: Final heatmap shape: {heatmap.shape}")
    return heatmap

def overlay_heatmap_on_frame(frame, heatmap):
    print(f"DEBUG: Overlaying heatmap - frame shape: {frame.shape}, heatmap shape: {heatmap.shape}")
    
    # Handle multi-dimensional heatmaps
    if heatmap.ndim > 2:
        print(f"DEBUG: Reducing heatmap from {heatmap.shape} to 2D")
        # Average across all extra dimensions to get 2D
        while heatmap.ndim > 2:
            heatmap = np.mean(heatmap, axis=0)
        print(f"DEBUG: Reduced heatmap shape: {heatmap.shape}")
    
    # Ensure heatmap is 2D
    if heatmap.ndim != 2:
        print(f"WARNING: Heatmap is not 2D: {heatmap.shape}")
        if heatmap.size == 0:
            return frame
        heatmap = heatmap.squeeze()
        if heatmap.ndim != 2:
            return frame
    
    # Ensure non-empty heatmap
    if heatmap.size == 0:
        print("WARNING: Empty heatmap")
        return frame
    
    print(f"DEBUG: Final heatmap shape for processing: {heatmap.shape}")
    
    # Resize heatmap to match frame dimensions
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (frame.shape[1], frame.shape[0]))
    
    # Apply colormap using matplotlib
    colormap = plt.cm.jet
    heatmap_colored = colormap(heatmap_resized)
    
    # Convert to uint8 and remove alpha channel if present
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Blend the images
    overlayed = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    return overlayed

def predict_video(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"DEBUG: Total frames in video: {total_frames}")

    buffer = []
    predictions = []
    visualizations = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_idx = 0
    clip_idx = 0

    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frames.append(frame)
        buffer.append(preprocess_frame(frame))

        if len(buffer) == clip_len:
            status_text.text(f"Processing clip {clip_idx + 1}...")
            
            clip = torch.stack(buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(clip)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
                
            print(f"DEBUG: Clip {clip_idx}: probability = {prob:.4f}")

            # Grad-CAM if violent
            if prob > 0.5:
                try:
                    print(f"DEBUG: Generating Grad-CAM for clip {clip_idx} (prob: {prob:.4f})")
                    heatmap = compute_gradcam(clip)
                    
                    if heatmap.size > 0:
                        selected_frames = [4, 8, 12]
                        vis = []
                        for idx in selected_frames:
                            if idx < len(raw_frames):
                                raw_frame = raw_frames[idx].copy()
                                overlayed = overlay_heatmap_on_frame(raw_frame, heatmap)
                                vis.append(overlayed)
                        
                        if vis:
                            visualizations.append((prob, vis))
                            print(f"DEBUG: Added visualization for clip {clip_idx}")
                    else:
                        print(f"DEBUG: Empty heatmap for clip {clip_idx}")
                        
                except Exception as e:
                    print(f"ERROR: Failed to generate Grad-CAM for clip {clip_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()

            buffer = buffer[stride:]
            raw_frames = raw_frames[stride:]

            clip_idx += 1
            progress = int(clip_idx * stride / total_frames * 100)
            progress_bar.progress(min(progress, 100))

        frame_idx += 1

    cap.release()
    status_text.text("Processing complete!")
    print(f"DEBUG: Total clips processed: {clip_idx}")
    print(f"DEBUG: Total predictions: {len(predictions)}")
    print(f"DEBUG: Total visualizations: {len(visualizations)}")
    
    return predictions, visualizations

# Streamlit UI
st.title("📹 R3D Violence Detection + Region Visualization")
st.write("Upload a video. The model scans clips and highlights violent regions using Grad-CAM.")

uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)

    if st.button("Run Detection + Grad-CAM"):
        with st.spinner("Analyzing..."):
            predictions, visualizations = predict_video(tfile.name)

        if len(predictions) == 0:
            st.error("No predictions generated. Check your video file.")
        else:
            avg_prob = np.mean(predictions)
            max_prob = np.max(predictions)
            clip_threshold = 0.5
            min_segments = max(1, int(0.05 * len(predictions)))
            violent_clips = [p for p in predictions if p > clip_threshold]

            st.write(f"**Results:**")
            st.write(f"- Average Probability: {avg_prob:.3f}")
            st.write(f"- Maximum Probability: {max_prob:.3f}")
            st.write(f"- Clips above threshold ({clip_threshold}): {len(violent_clips)} / {len(predictions)}")

            if len(violent_clips) >= min_segments:
                st.error(f"⚠️ Violence Detected!")
            else:
                st.success("✅ No violence detected.")

            # Show prediction chart
            st.subheader("📊 Prediction Timeline")
            st.line_chart(predictions)
            
            # Show visualizations
            st.subheader("🧠 Grad-CAM Visualizations")
            
            if len(visualizations) == 0:
                st.info("No Grad-CAM visualizations generated (no clips above threshold or visualization failed)")
            else:
                for i, (prob, frames) in enumerate(visualizations):
                    st.write(f"**Clip {i+1}** - Violence Probability: {prob:.3f}")
                    cols = st.columns(len(frames))
                    for j, frame in enumerate(frames):
                        with cols[j]:
                            st.image(frame, channels="RGB", caption=f"Frame {j+1}")