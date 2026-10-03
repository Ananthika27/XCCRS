import sys
import os
import time

# Ensure modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.reasoning_engine import ReasoningEngine
from modules.output_generator import OutputGenerator
from modules.knowledge_base import KnowledgeBase

def run_reasoning_demo(scene_type, detections, person_attributes, case_name):
    """Refactored helper to run a specific case."""
    print(f"\n--- RUNNING CASE: {case_name} ---")
    
    # Load rules dynamically
    kb = KnowledgeBase("configs/safety_rules.json")
    requirements = kb.get_scene_requirements(scene_type)
    
    # Run Engine
    engine = ReasoningEngine()
    result = engine.analyze_compliance(scene_type, detections, person_attributes, requirements)
    
    # Generate Report
    gen = OutputGenerator()
    print(gen.generate_report(result))
    
    input("\nPress Enter to continue to menu...")

def example_factory_compliant():
    scene = "factory"
    dets = [{"id": 1, "label": "person", "conf": 0.9}]
    attrs = {1: {"helmet": True, "gloves": True, "safety_vest": True}}
    run_reasoning_demo(scene, dets, attrs, "Factory - Fully Compliant Worker")

def example_factory_violation():
    scene = "factory"
    dets = [{"id": 1, "label": "person", "conf": 0.9}]
    attrs = {1: {"helmet": False, "gloves": True, "safety_vest": True}} # No Helmet
    run_reasoning_demo(scene, dets, attrs, "Factory - Missing Helmet")

def example_warehouse_hazard():
    scene = "warehouse"
    dets = [
        {"id": 1, "label": "person", "conf": 0.9},
        {"id": 2, "label": "spilled_liquid", "conf": 0.85} # Hazard!
    ]
    attrs = {1: {"safety_vest": True, "steel_toe_boots": True}}
    run_reasoning_demo(scene, dets, attrs, "Warehouse - Spilled Liquid Check")

def example_construction_critical():
    scene = "construction"
    dets = [{"id": 1, "label": "person", "conf": 0.95}]
    attrs = {1: {"helmet": True, "high_vis_vest": True, "harness": False}} # Missing harness!
    run_reasoning_demo(scene, dets, attrs, "Construction - Missing Harness (Critical)")

def example_lab_mixed():
    scene = "laboratory"
    dets = [
        {"id": 1, "label": "person", "conf": 0.9},
        {"id": 2, "label": "person", "conf": 0.92}
    ]
    attrs = {
        1: {"lab_coat": True, "safety_goggles": True, "gloves": True}, # Safe
        2: {"lab_coat": True, "safety_goggles": False, "gloves": False} # Unsafe
    }
    run_reasoning_demo(scene, dets, attrs, "Lab - Mixed Compliance")

def example_unknown_scene():
    scene = "office" # Not in rules
    dets = [{"id": 1, "label": "person", "conf": 0.9}]
    attrs = {1: {}}
    run_reasoning_demo(scene, dets, attrs, "Unknown Scene Handling")

def example_fire_emergency():
    scene = "factory"
    dets = [
        {"id": 1, "label": "person", "conf": 0.9},
        {"id": 2, "label": "fire", "conf": 0.99}
    ]
    attrs = {1: {"helmet": True, "gloves": True, "safety_vest": True}}
    run_reasoning_demo(scene, dets, attrs, "Factory - FIRE EMERGENCY")

def main_menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=======================================================")
        print("   XCCRS SYNTHETIC REASONING EXAMPLE SUITE             ")
        print("=======================================================")
        print("Select a scenario to verify the Neuro-Symbolic Logic:")
        print("1. Factory - Fully Compliant")
        print("2. Factory - PPE Violation (No Helmet)")
        print("3. Warehouse - Environmental Hazard")
        print("4. Construction - Critical Safety Violation")
        print("5. Laboratory - Multi-Person Mixed Compliance")
        print("6. Unknown Environment Handling")
        print("7. Factory - CRITICAL EMERGENCY (Fire)")
        print("q. Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == '1': example_factory_compliant()
        elif choice == '2': example_factory_violation()
        elif choice == '3': example_warehouse_hazard()
        elif choice == '4': example_construction_critical()
        elif choice == '5': example_lab_mixed()
        elif choice == '6': example_unknown_scene()
        elif choice == '7': example_fire_emergency()
        elif choice == 'q': 
            print("Exiting.")
            break
        else:
            print("Invalid choice, try again.")
            time.sleep(1)

if __name__ == "__main__":
    main_menu()
