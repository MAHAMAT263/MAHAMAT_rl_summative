import os
import json
import csv

def save_hyperparameters(hparams, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(hparams, f, indent=4)

def init_csv_logger(csv_path, headers):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def append_to_csv(csv_path, row):
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)