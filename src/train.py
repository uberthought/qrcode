from data_loader import QRCodeDataset, CHARSET, MAX_TEXT_LEN
from model import create_model, get_best_device, save_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import QRCodeSimpleNet

def train():
    """
    Trains the QR code image-to-text model using per-character CrossEntropyLoss.
    """

    # Create or load model
    model_path = 'qrcode_recognizer.pt'
    device = get_best_device()
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = QRCodeSimpleNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"Loaded model from {model_path} on device: {device}")
    else:
        print("Creating new model...")
        model = create_model()
        model.to(device)
        print(f"Created new model on device: {device}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # create validation data
    val_dataset = QRCodeDataset(num_samples=400, force_synthetic=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    # Training loop
    print("Training model...")
    epochs = 256
    for epoch in range(epochs):
        # create train data
        print(f"Epoch {epoch+1}/{epochs}", end='', flush=True)

        print(" - Creating data..." , end='', flush=True)
        train_dataset = QRCodeDataset(num_samples=2048, force_synthetic=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

        print(" - Training...", end='', flush=True)
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            imgs, labels, _ = batch
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)  # (B, seq_len, charset_size)
            logits = logits.view(-1, logits.size(-1))  # (B*seq_len, charset_size)
            labels = labels.view(-1)  # (B*seq_len)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f" - Loss: {epoch_loss:.4f}")

        # Validation
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch in val_loader:
        #         imgs, labels, _ = batch
        #         imgs, labels = imgs.to(device), labels.to(device)
        #         logits = model(imgs)
        #         logits = logits.view(-1, logits.size(-1))
        #         labels = labels.view(-1)
        #         loss = criterion(logits, labels)
        #         val_loss += loss.item() * imgs.size(0)
        # val_loss /= len(val_loader.dataset)
        # print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Save the trained model
    print("Saving model...")
    save_model(model, 'qrcode_recognizer.pt')
    print("Model saved as qrcode_recognizer.pt")

if __name__ == '__main__':
    train()
