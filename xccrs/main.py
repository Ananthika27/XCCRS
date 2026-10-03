import argparse
import logging
import os
import sys

# Ensure modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.input_handler import InputHandler
from modules.scene_understanding import SceneUnderstanding
from modules.knowledge_base import KnowledgeBase
from modules.object_detection import ObjectDetector
from modules.attribute_verification import AttributeVerifier
from modules.reasoning_engine import ReasoningEngine
from modules.output_generator import OutputGenerator
from modules.action_simulator import ActionSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XCCRS_Main")

def main():
    parser = argparse.ArgumentParser(description="XCCRS: Vision-Language Cognitive Reasoning for Safety")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--rules", default="configs/safety_rules.json", help="Path to safety rules JSON")
    args = parser.parse_args()

    # 1. Initialize Modules
    logger.info("Initializing XCCRS modules...")
    input_handler = InputHandler()
    scene_model = SceneUnderstanding()
    kb = KnowledgeBase(args.rules)
    detector = ObjectDetector("yolov8n.pt")
    verifier = AttributeVerifier()
    reasoner = ReasoningEngine()
    output_gen = OutputGenerator()
    simulator = ActionSimulator()

    # 2. Process Input
    frames = input_handler.process_input(args.input)
    if not frames:
        logger.error("No frames processed.")
        return

    # In a real system, we'd loop. For this demo, detailed report on FIRST frame usually, 
    # but we can do a loop. Let's do the first frame for the elaborate report, 
    # or loop and print summaries.
    # The requirement implication is "Accept image OR video...".
    # We will process the first frame fully for the "Report" format validation.
    
    frame = frames[0] 
    logger.info("Analyzing first frame...")

    # 3. Scene Understanding
    scene_type, caption = scene_model.analyze_scene(frame)
    logger.info(f"Scene detected: {scene_type} ({caption})")

    # 4. Knowledge Base Retrieval
    requirements = kb.get_scene_requirements(scene_type)
    if not requirements['rules']:
        logger.warning(f"No rules found for scene type '{scene_type}'. Proceeding with caution.")

    # 5. Object Detection
    detections = detector.detect(frame)
    logger.info(f"Detections: {[d['label'] for d in detections]}")

    # 6. Attribute Verification
    # We need to check PPE for detected persons.
    mandatory_ppe = requirements.get("mandatory_ppe", [])
    person_attributes = {} # {id: {attr: bool}}

    for det in detections:
        if det['label'] == 'person':
            pid = det['id']
            # Crop and verify
            # We check ALL mandatory PPE for the scene on EACH person
            if mandatory_ppe:
                # pass the whole list of attributes to check at once to save overhead if implemented that way, 
                # but our verifier does a loop.
                # Box is [x1, y1, x2, y2]
                res = verifier.verify_attributes(frame, det['box'], mandatory_ppe)
                person_attributes[pid] = res

    # 7. Reasoning
    analysis_result = reasoner.analyze_compliance(
        scene_type=scene_type,
        detections=detections,
        person_attributes=person_attributes,
        requirements=requirements
    )

    # 8. Output Generation
    report = output_gen.generate_report(analysis_result)
    print("\n" + report + "\n")

    # 9. Action Simulation
    simulator.execute_actions(analysis_result)

if __name__ == "__main__":
    main()
