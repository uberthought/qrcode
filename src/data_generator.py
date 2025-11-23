import os
import csv
import numpy as np
import cv2
from tqdm import tqdm
from data_loader import CHARSET, generate_qrcode_image, MAX_TEXT_LEN

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_SAMPLES = 1000
IMG_SIZE = 128
CSV_FILENAME = 'labels.csv'

os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, CSV_FILENAME)

with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'text'])
    for i in tqdm(range(NUM_SAMPLES), desc='Generating QR dataset'):
        text_len = np.random.randint(4, MAX_TEXT_LEN+1)
        text = ''.join(np.random.choice(list(CHARSET), text_len))
        img_np = generate_qrcode_image(text)
        img_filename = f'qrcode_{i:05d}.png'
        img_path = os.path.join(DATA_DIR, img_filename)
        cv2.imwrite(img_path, img_np)
        writer.writerow([img_filename, text])

print(f"Generated {NUM_SAMPLES} samples in {DATA_DIR}")
