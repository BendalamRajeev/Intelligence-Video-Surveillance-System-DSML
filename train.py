# train.py
import torch
from torch.utils.data import DataLoader
from dataset import ClipDataset
from models.three_d_cnn_model import R3DViolenceDetector
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(epochs=20, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = R3DViolenceDetector(pretrained=True).to(device)
    train_set = ClipDataset("dataset/train", clip_len=16)
    print(f"Loaded {len(train_set)} training samples.")
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for clips, labels in progress:
            # clips: [B, C, T, H, W] ✅ already correct
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        acc = correct / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")
        loss_history.append(total_loss)
        acc_history.append(acc)

    torch.save(model.state_dict(), "r3d_model.pth")
    print("Training complete. Model saved as r3d_model.pth")
    return loss_history, acc_history

if __name__ == "__main__":
    loss, acc = train_model()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(acc, label='Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.tight_layout()
    plt.savefig("r3d_training_graph.png")
    plt.show()
