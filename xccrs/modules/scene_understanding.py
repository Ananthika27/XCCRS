import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

class SceneUnderstanding:
    """
    Uses BLIP to generate captions for images and maps them to scene types.
    Strictly deterministic mapping based on keywords.
    """
    
    SCENE_KEYWORDS = {
        "factory": ["factory", "machine", "industrial", "plant", "manufacturing", "robot", "assembly", "conveyor"],
        "warehouse": ["warehouse", "storage", "boxes", "shelves", "forklift", "pallet", "inventory"],
        "construction": ["construction", "site", "building", "crane", "scaffold", "excavator", "hardhat"],
        "laboratory": ["laboratory", "lab", "scientist", "chemicals", "microscope", "beaker", "test tube"]
    }

    def __init__(self):
        logger.info("Loading BLIP model for scene understanding...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            # Check for local model first to perform offline loading
            import os
            local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/blip-base")
            
            if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
                logger.info(f"Loading BLIP from local path: {local_path}")
                model_name_or_path = local_path
            else:
                logger.info("Local model not found, downloading from Hugging Face...")
                model_name_or_path = "Salesforce/blip-image-captioning-base"

            self.processor = BlipProcessor.from_pretrained(model_name_or_path)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
            self.model.eval()
            logger.info("BLIP model loaded successfully.")
            self.mock_mode = False
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            logger.warning("SWITCHING TO MOCK MODE for Scene Understanding.")
            self.mock_mode = True

    def analyze_scene(self, image: Image.Image) -> Tuple[str, str]:
        """
        Generates a caption and determines the scene type.
        """
        if self.mock_mode:
            logger.warning("Using Mock Scene Analysis")
            return "factory", "Mock Caption: Industrial factory setting (Simulated due to model load failure)"
            
        caption = self._generate_caption(image)
        scene_type = self._classify_scene(caption)
        return scene_type, caption

    def _generate_caption(self, image: Image.Image) -> str:
        """Runs BLIP inference to caption the image."""
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Generated caption: {caption}")
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "unknown scene"

    def _classify_scene(self, caption: str) -> str:
        """
        Maps caption keywords to scene types.
        Returns 'unknown' if no keywords match.
        """
        caption_lower = caption.lower()
        
        # Check against keywords
        scores = {scene: 0 for scene in self.SCENE_KEYWORDS}
        
        for scene, keywords in self.SCENE_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', caption_lower):
                    scores[scene] += 1
        
        # Get scene with highest score, match must be > 0
        best_scene = max(scores, key=scores.get)
        if scores[best_scene] > 0:
            logger.info(f"Classified scene as: {best_scene} (score: {scores[best_scene]})")
            return best_scene
        
        logger.warning("Could not classify scene type from caption.")
        return "unknown"

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    try:
        scene_model = SceneUnderstanding()
        # img = Image.open("factory_test.jpg")
        # print(scene_model.analyze_scene(img))
    except Exception as e:
        print(f"Init failed (expected during build if no internet/files): {e}")
