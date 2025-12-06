
import torch
from data_loader import generate_qrcode_image, decode_prediction, CHARSET, MAX_TEXT_LEN
import numpy as np
from torchvision import transforms
from model import load_model


def main():
    # Load model
    model = load_model('qrcode_recognizer.pt')

    # Test data

    test_strings = [
        "Hello, world!",
        "https://github.com",
        "1234567890",
        "Test QR Code",
        "OpenAI GPT-4",
        "Python is fun!",
        "Scan me please",
        "Contact: 555-1234",
        "user@example.com",
        "https://openai.com",
        "42 is the answer",
        "Sample QR Data",
        "AI Revolution",
        "Data Science 101",
        "Machine Learning",
        "Deep Learning",
        "Neural Networks",
        "Test String 001",
        "Test String 002",
        "Test String 003",
        "Test String 004",
        "Test String 005",
        "The quick brown fox",
        "jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "Grumpy wizards make toxic brew for the evil queen and jack.",
        "123 Main Street",
        "Apt. 4B",
        "New York, NY 10001",
        "2025-12-03",
        "12:34:56",
        "abcdefg",
        "ABCDEFG",
        "AaBbCcDdEeFf",
        "QWERTYUIOP",
        "asdfghjkl",
        "zxcvbnm",
        "0987654321",
        "https://example.org",
        "https://test.com",
        "https://bit.ly/3xyz",
        "https://tinyurl.com/abc123",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://en.wikipedia.org/wiki/QR_code",
        "https://docs.python.org/3/",
        "https://pytorch.org/",
        "https://numpy.org/",
        "https://scikit-learn.org/",
        "https://huggingface.co/",
        "https://stackoverflow.com/",
        "https://reddit.com/r/MachineLearning/",
        "https://twitter.com/",
        "https://facebook.com/",
        "https://linkedin.com/",
        "https://instagram.com/",
        "https://t.me/qrtest",
        "https://discord.gg/abcdef",
        "https://slack.com/",
        "https://zoom.us/",
        "https://meet.google.com/",
        "https://calendar.google.com/",
        "https://mail.google.com/",
        "https://drive.google.com/",
        "https://dropbox.com/",
        "https://box.com/",
        "https://icloud.com/",
        "https://one.microsoft.com/",
        "https://github.com/uberthought/qrcode",
        "https://github.com/openai/gpt-4",
        "https://github.com/pytorch/pytorch",
        "https://github.com/numpy/numpy",
        "https://github.com/scikit-learn/scikit-learn",
        "https://github.com/huggingface/transformers",
        "https://github.com/keras-team/keras",
        "https://github.com/tensorflow/tensorflow",
        "https://github.com/psf/requests",
        "https://github.com/pallets/flask",
        "https://github.com/django/django",
        "https://github.com/tiangolo/fastapi",
        "https://github.com/matplotlib/matplotlib",
        "https://github.com/pandas-dev/pandas",
        "https://github.com/pytest-dev/pytest",
        "https://github.com/encode/starlette",
        "https://github.com/plotly/plotly.py",
        "https://github.com/streamlit/streamlit",
        "https://github.com/ultralytics/yolov5",
        "https://github.com/opencv/opencv",
        "https://github.com/psf/black",
        "https://github.com/psf/pep8",
        "https://github.com/psf/pycodestyle",
        "https://github.com/psf/pyflakes",
        "https://github.com/psf/isort",
        "https://github.com/psf/pylint",
        "https://github.com/psf/mypy",
        "https://github.com/psf/pyright",
        "https://github.com/psf/pytype",
        "https://github.com/psf/pyre-check",
        "https://github.com/psf/py-spy",
        "https://github.com/psf/pyinstrument",
        "https://github.com/psf/pyperf",
        "https://github.com/psf/pycparser",
        "https://github.com/psf/pyparsing",
        "https://github.com/psf/pyyaml",
        "https://github.com/psf/pytest-cov"
    ]

    correct = 0
    total = len(test_strings)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = model.conv1.weight.device

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
