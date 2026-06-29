"""
Download RSNA Breast Cancer Detection competition data to G drive.

kagglehub respects KAGGLE_CACHE_DIR for the download destination.
Requires:
  - pip install kagglehub
  - Kaggle API credentials: ~/.kaggle/kaggle.json  (or KAGGLE_USERNAME + KAGGLE_KEY env vars)
    Get from: https://www.kaggle.com/settings -> API -> Create New Token

Usage:
    python scripts/download_rsna.py
"""
import os
import sys

# Route download to G drive BEFORE importing kagglehub (it reads env at import time)
RSNA_ROOT = r"G:\breast-cancer-research\rsna"
os.environ["KAGGLE_CACHE_DIR"] = RSNA_ROOT
os.makedirs(RSNA_ROOT, exist_ok=True)

# Set token via env var (works alongside ~/.kaggle/access_token)
TOKEN_FILE = os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")
if os.path.exists(TOKEN_FILE):
    token = open(TOKEN_FILE).read().strip()
    os.environ["KAGGLE_API_TOKEN"] = token
    print(f"Token loaded from {TOKEN_FILE}")
else:
    print("WARNING: No token file found at ~/.kaggle/access_token")

import kagglehub

print(f"Downloading RSNA Breast Cancer Detection dataset to: {RSNA_ROOT}")
print("This is ~280 GB — expect several hours depending on connection speed.\n")

path = kagglehub.competition_download("rsna-breast-cancer-detection")

print(f"\nPath to competition files: {path}")
