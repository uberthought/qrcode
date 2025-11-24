import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import CHARSET, MAX_TEXT_LEN


def save_model(model, path):
    """Save model state dict to file."""
    torch.save(model.state_dict(), path)

def load_model(path, device=None):
    """Load model state dict from file and return model on device (if given)."""
    model = QRCodeSimpleNet()  # Use the correct model class
    if device is None:
        device = get_best_device()
    print(f"Loading model on device: {device}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_best_device():
    """Return best available torch.device: MPS (Mac), else CUDA, else CPU."""
    # Apple Silicon (MPS) troubleshooting: ensure torch version >= 1.12 and MPS is available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

CHARSET_SIZE = len(CHARSET)
SEQ_LEN = MAX_TEXT_LEN  # Fixed output length matches label length


# Simple CNN for fixed-length sequence classification (no CTC)
class QRCodeSimpleNet(nn.Module):
    def __init__(self, charset_size=CHARSET_SIZE, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, seq_len * charset_size)

    def forward(self, x):
        # x: (B, 1, 128, 128)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.classifier(x)     # (B, seq_len * charset_size)
        x = x.view(x.size(0), self.seq_len, -1)  # (B, seq_len, charset_size)
        # During inference, you can get predicted indices with x.argmax(-1)
        return x

def create_model():
    """Create a QRCodeSimpleNet model and move it to the best device."""
    model = QRCodeSimpleNet()
    return model
