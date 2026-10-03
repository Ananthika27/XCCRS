import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Loads and serves deterministic safety rules from a JSON file.
    Acts as the symbolic truth source for the system.
    """
    
    def __init__(self, rules_path: str = "configs/safety_rules.json"):
        # Resolve absolute path relative to this file if not absolute
        if not os.path.isabs(rules_path):
             # Assume rules_path is relative to the project root (parent of modules)
             current_dir = os.path.dirname(os.path.abspath(__file__))
             project_root = os.path.dirname(current_dir)
             rules_path = os.path.join(project_root, rules_path)

        self.rules_path = rules_path
        self.rules_data = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        """Loads safety rules from JSON."""
        if not os.path.exists(self.rules_path):
            logger.error(f"Safety rules file not found: {self.rules_path}")
            raise FileNotFoundError(f"Safety rules file missing: {self.rules_path}")
            
        try:
            with open(self.rules_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded safety rules for {len(data)} scene types.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in safety rules: {e}")
            raise

    def get_scene_requirements(self, scene_type: str) -> Dict[str, Any]:
        """
        Retrieves rules and requirements for a specific scene type.
        
        Args:
            scene_type: 'factory', 'warehouse', etc.
            
        Returns:
            Dict containing mandatory_ppe, danger_signals, and rules.
            Returns empty dict structure if scene unknown.
        """
        data = self.rules_data.get(scene_type)
        if not data:
            logger.warning(f"No safety rules found for scene type: {scene_type}")
            return {
                "mandatory_ppe": [],
                "danger_signals": [],
                "rules": []
            }
        return data

    def get_all_scene_types(self) -> List[str]:
        return list(self.rules_data.keys())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb = KnowledgeBase()
    print(kb.get_scene_requirements("factory"))
