from torchinfo import summary
from models.three_d_cnn_model import R3DViolenceDetector
import torch

model = R3DViolenceDetector(pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print summary like Keras
summary(model, input_size=(1, 3, 16, 112, 112))
