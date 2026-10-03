import os
import sys
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import clip
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

# Set timeout to 10 minutes (600 seconds) to handle slow connections for large models
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

def download_blip():
    logger.info("Downloading BLIP model (Salesforce/blip-image-captioning-base)...")
    try:
        # forcing download
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", force_download=False)
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", force_download=False)
        logger.info("BLIP download complete.")
    except Exception as e:
        logger.error(f"Failed to download BLIP: {e}")
        sys.exit(1)

def download_yolo():
    logger.info("Downloading YOLOv8n model...")
    try:
        # explicit download if needed, but YOLO class usually handles it. 
        # accessing it triggers download.
        model = YOLO("yolov8n.pt")
        logger.info("YOLO download complete.")
    except Exception as e:
        logger.error(f"Failed to download YOLO: {e}")
        sys.exit(1)

def download_clip():
    logger.info("Downloading CLIP model (ViT-B/32)...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip.load("ViT-B/32", device=device)
        logger.info("CLIP download complete.")
    except Exception as e:
        logger.error(f"Failed to download CLIP: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("--- XCCRS Model Downloader ---")
    print("Pre-fetching models to avoid timeouts during app startup.")
    
    download_blip()
    download_yolo()
    download_clip()
    
    print("\nAll models downloaded successfully!")
    print("You can now run: streamlit run xccrs/demo_app.py")
