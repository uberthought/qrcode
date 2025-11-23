


# PyTorch Dataset for QR code image-to-text pairs
import os
import cv2
import numpy as np
import qrcode
import string
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Character set: numbers, uppercase, lowercase, and a few symbols (can be expanded)
CHARSET = string.ascii_letters + string.digits + ' .,-_:/@'
CHAR2IDX = {c: i+1 for i, c in enumerate(CHARSET)}  # 0 is reserved for padding
IDX2CHAR = {i+1: c for i, c in enumerate(CHARSET)}
PAD_IDX = 0
MAX_TEXT_LEN = 32

def text_to_indices(text, max_length=MAX_TEXT_LEN):
    indices = [CHAR2IDX.get(c, PAD_IDX) for c in text[:max_length]]
    indices += [PAD_IDX] * (max_length - len(indices))
    return torch.tensor(indices, dtype=torch.long)

def indices_to_text(indices):
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    chars = [IDX2CHAR.get(int(idx), '') for idx in indices if idx != PAD_IDX]
    return ''.join(chars)

def generate_qrcode_image(text):
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    img_pil = qr.make_image(fill_color="black", back_color="white")
    img_np = np.array(img_pil.convert('L'))
    img_np = cv2.resize(img_np, (128, 128))
    return img_np

class QRCodeDataset(Dataset):
    """
    PyTorch Dataset for (image, text) QR code pairs.
    If data_dir is None, generates synthetic QR codes with random text.
    Otherwise, expects data_dir to contain images and a CSV with text labels.
    """
    def __init__(self, data_dir=None, num_samples=1000, transform=None, force_synthetic=False):
        self.data_dir = data_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.samples = []

        if data_dir is None or force_synthetic:
            # Generate synthetic QR codes
            for _ in range(num_samples):
                text_len = np.random.randint(4, MAX_TEXT_LEN+1)
                text = ''.join(np.random.choice(list(CHARSET), text_len))
                img_np = generate_qrcode_image(text)
                self.samples.append((img_np, text))
        else:
            # Load images and labels from a CSV file in data_dir
            import csv
            csv_path = os.path.join(data_dir, 'labels.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                for row in reader:
                    if len(row) < 2:
                        continue
                    img_filename, text = row[0], row[1]
                    img_path = os.path.join(data_dir, img_filename)
                    if not os.path.exists(img_path):
                        continue
                    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_np is None:
                        continue
                    img_np = cv2.resize(img_np, (128, 128))
                    self.samples.append((img_np, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, text = self.samples[idx]
        img = self.transform(img_np)
        label = text_to_indices(text)
        return img, label, text

# Utility for external use
def decode_prediction(indices):
    return indices_to_text(indices)
