
import cv2
import torch
from data_loader import decode_prediction
from torchvision import transforms



def predict_image(image_path, model_path='qrcode_recognizer.pt'):
    """
    Predicts the text content of a QR code image using the trained model.
    """
    # Load the model
    from model import load_model
    model = load_model(model_path)

    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    img = cv2.resize(img, (128, 128))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 1, 128, 128)
    with torch.no_grad():
        logits = model(img_tensor)  # (1, seq_len, charset_size)
        pred_indices = logits.argmax(-1)[0]  # (seq_len,)
        pred_text = decode_prediction(pred_indices)
    print(f"Predicted text: {pred_text}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict text from a QR code image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    args = parser.parse_args()
    predict_image(args.image_path)
