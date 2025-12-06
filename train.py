from data_loader import QRCodeDataset, CHARSET, MAX_TEXT_LEN
from model import create_model, get_best_device, save_model, load_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def train():
    """
    Trains the QR code image-to-text model using per-character CrossEntropyLoss.
    """

    # Create or load model
    model_path = 'qrcode_recognizer.pt'
    device = get_best_device()
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model()

    # Training loop
    print("Training model...")
    epochs = 16

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=64, epochs=epochs)

    # Create training data loader once
    train_dataset = QRCodeDataset(num_samples=512, force_synthetic=True)
    # pin_memory is not supported on MPS
    use_pin_memory = device.type != 'mps'
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=use_pin_memory, persistent_workers=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}", end='', flush=True)
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
            scheduler.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f" - Loss: {epoch_loss:.4f}")

    # Save the trained model
    print("Saving model...")
    save_model(model, 'qrcode_recognizer.pt')
    print("Model saved as qrcode_recognizer.pt")

if __name__ == '__main__':
    train()
