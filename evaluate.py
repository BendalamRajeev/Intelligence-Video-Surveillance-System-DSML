# evaluate.py

import torch
from torch.utils.data import DataLoader
from dataset import ClipDataset
from models.three_d_cnn_model import R3DViolenceDetector
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = R3DViolenceDetector(pretrained=False).to(device)
    model.load_state_dict(torch.load("r3d_model.pth", map_location=device))
    model.eval()

    # Load test data
    test_set = ClipDataset("dataset/test", clip_len=16)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc="Evaluating"):
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\\n✅ Evaluation Results:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # ✅ Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Violent', 'Violent'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix as 'confusion_matrix.png'.")

    # ✅ Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'Loss': avg_loss,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }])
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    print("Saved metrics to 'evaluation_metrics.csv'.")

    return avg_loss, acc, precision, recall, f1

if __name__ == "__main__":
    evaluate_model()
