
import torch
from data_loader import generate_qrcode_image, decode_prediction, CHARSET, MAX_TEXT_LEN
import numpy as np
from torchvision import transforms


def main():
    # Load model
    from model import load_model, get_best_device
    device = get_best_device()
    model = load_model('qrcode_recognizer.pt', device=device)

    # Test data
    test_strings = [
        "Hello, world!",
        "https://github.com",
        "1234567890",
        "Test QR Code",
        "OpenAI GPT-4"
    ]
    correct = 0
    total = len(test_strings)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for s in test_strings:
        img_np = generate_qrcode_image(s)
        img = transform(img_np).unsqueeze(0)  # (1, 1, 128, 128)
        img = img.to(device)
        with torch.no_grad():
            logits = model(img)  # (1, seq_len, charset_size)
            pred_indices = logits.argmax(-1)[0]  # (seq_len,)
            pred_text = decode_prediction(pred_indices)
        print(f"Input: {s}\nPredicted: {pred_text}\n")
        if pred_text == s:
            correct += 1
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

if __name__ == "__main__":
    main()
