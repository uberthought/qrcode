
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

# Determine the last used index from the existing CSV file
last_index = -1
if os.path.exists(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0].startswith('qrcode_') and row[0].endswith('.png'):
                try:
                    idx = int(row[0][7:12])
                    if idx > last_index:
                        last_index = idx
                except ValueError:
                    continue

# Open CSV in append mode if it exists, else write mode
write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
mode = 'a' if not write_header else 'w'
with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(['filename', 'text'])
    for i in tqdm(range(last_index + 1, last_index + 1 + NUM_SAMPLES), desc='Generating QR dataset'):
        text_len = np.random.randint(4, MAX_TEXT_LEN+1)
        text = ''.join(np.random.choice(list(CHARSET), text_len))
        img_filename = f'qrcode_{i:05d}.png'
        img_path = os.path.join(DATA_DIR, img_filename)
        img_np = generate_qrcode_image(text)
        cv2.imwrite(img_path, img_np)
        writer.writerow([img_filename, text])

print(f"Generated {NUM_SAMPLES} new samples in {DATA_DIR}, continuing from index {last_index+1}")
