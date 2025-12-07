import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import CHARSET, MAX_TEXT_LEN

def create_model():
    """Create and return a new QRCodeResNet model instance."""
    model = QRCodeResNet()
    device = get_best_device()
    model.to(device)
    return model

def save_model(model, path):
    """Save model state dict to file."""
    torch.save(model.state_dict(), path)

def load_model(path):
    """Load model state dict from file and return model on best device."""
    model = QRCodeResNet()  # Use the new ResNet-like model class
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

# Basic Residual Block for ResNet-like architecture
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ResNet-like network for QR code recognition
class QRCodeResNet(nn.Module):
    def __init__(self, charset_size=CHARSET_SIZE, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(32, 32, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=3, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=3, stride=2)
        self.classifier = nn.Linear(256 * 8 * 8, seq_len * charset_size)
        self.to(get_best_device())

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (, 1, 128, 128)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(x.size(0), self.seq_len, -1)
       # x: (, seq_len, charset_size)
        return x
