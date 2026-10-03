import cv2
from PIL import Image
import os
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class InputHandler:
    """
    Handles ingestion of images or videos.
    Converts inputs into a list of PIL Images for downstream processing.
    """
    
    def __init__(self):
        pass

    def process_input(self, source_path: str) -> List[Image.Image]:
        """
        Reads an image or video file and returns a list of PIL Images.
        
        Args:
            source_path: Path to the input image or video.
            
        Returns:
            List[Image.Image]: List of frames (1 for image, N for video).
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Input file not found: {source_path}")
            
        ext = os.path.splitext(source_path)[1].lower()
        
        # Image extensions
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            return self._process_image(source_path)
            
        # Video extensions
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return self._process_video(source_path)
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _process_image(self, path: str) -> List[Image.Image]:
        """Loads a single image."""
        logger.info(f"Processing image: {path}")
        try:
            img = Image.open(path).convert("RGB")
            return [img]
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def _process_video(self, path: str, sample_rate: int = 1) -> List[Image.Image]:
        """
        Process video and sample frames at  Hz (frames per second).
        
        Args:
            path: Video file path.
            sample_rate: Number of frames to extract per second.
            
        Returns:
            List[Image.Image]: Extracted frames.
        """
        logger.info(f"Processing video: {path}")
        frames = []
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning("Could not determine FPS, defaulting to 30")
            fps = 30
            
        # Calculate frame interval to achieve desired sample rate
        # e.g., if FPS=30 and sample_rate=1, we need every 30th frame
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
            
        frame_idx = 0
        extracted_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            if frame_idx % interval == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                frames.append(pil_img)
                extracted_count += 1
                
            frame_idx += 1
            
        cap.release()
        logger.info(f"Extracted {extracted_count} frames from video.")
        return frames

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    handler = InputHandler()
    # print(handler.process_input("test.jpg"))
