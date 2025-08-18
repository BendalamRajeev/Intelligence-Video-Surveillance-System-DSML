# dataset.py
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ClipDataset(Dataset):
    def __init__(self, root_dir, clip_len=16):
        self.samples = []
        self.clip_len = clip_len
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        for label_name in ['violent', 'non_violent']:
            label = 1 if label_name == 'violent' else 0
            label_path = os.path.join(root_dir, label_name)
            for video_folder in os.listdir(label_path):
                folder_path = os.path.join(label_path, video_folder)
                frames = sorted(os.listdir(folder_path))
                if len(frames) >= clip_len:
                    self.samples.append((folder_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        frame_files = sorted(os.listdir(folder_path))[:self.clip_len]
        frames = []
        for f in frame_files:
            img = Image.open(os.path.join(folder_path, f)).convert('RGB')
            img = self.transform(img)
            frames.append(img)
        clip = torch.stack(frames)  # [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W] ✅
        return clip, torch.tensor(label, dtype=torch.float32)
