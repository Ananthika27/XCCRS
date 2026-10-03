from ultralytics import YOLO
import logging
from PIL import Image
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Wraps YOLOv8 to detect objects and hazards.
    Normalizes detection labels to a standard safety schema.
    """
    
    # Mapping COCO labels to our safety ontology
    # COCO classes: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
    LABEL_MAP = {
        0: 'person',
        1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', # Vehicles
        56: 'chair', 63: 'laptop', # Equipment
        # Standard YOLOv8n has limited industrial classes, so we rely on 'person' mostly
        # and simple mappings.
    }

    def __init__(self, model_name: str = "yolov8n.pt"):
        logger.info(f"Loading YOLO model: {model_name}")
        try:
            self.model = YOLO(model_name)
            logger.info("YOLO model loaded.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Runs detection on an image.
        
        Returns:
            List of dicts: {'label': str, 'box': [x1, y1, x2, y2], 'conf': float}
        """
        results = self.model(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                # Default to normalized COCO label if not in our rigorous map
                label = self.model.names[cls_id]
                
                # Apply normalization if needed
                if cls_id == 0:
                    label = "person"
                
                # Filter low confidence
                if conf < 0.3:
                    continue

                detections.append({
                    "label": label,
                    "box": xyxy,
                    "conf": conf,
                    "id": cls_id # Keep original ID just in case
                })
        
        logger.info(f"YOLO detected {len(detections)} objects.")
        return detections

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        det = ObjectDetector()
        # img = Image.open("test.jpg")
        # print(det.detect(img))
    except Exception as e:
        print(f"Init failed (expected if no model downloaded): {e}")
