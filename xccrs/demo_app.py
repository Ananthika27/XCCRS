import streamlit as st
import os
import sys
from PIL import Image
import tempfile

# Ensure modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.scene_understanding import SceneUnderstanding
from modules.knowledge_base import KnowledgeBase
from modules.object_detection import ObjectDetector
from modules.attribute_verification import AttributeVerifier
from modules.reasoning_engine import ReasoningEngine
from modules.output_generator import OutputGenerator

st.set_page_config(page_title="XCCRS Safety System", layout="wide")

@st.cache_resource
def load_models():
    # Load all models once
    scene_model = SceneUnderstanding()
    kb = KnowledgeBase("configs/safety_rules.json")
    detector = ObjectDetector("yolov8n.pt")
    verifier = AttributeVerifier()
    reasoner = ReasoningEngine()
    output_gen = OutputGenerator()
    return scene_model, kb, detector, verifier, reasoner, output_gen

st.title("XCCRS: Neuro-Symbolic Safety Compliance")
st.markdown("### Vision-Language Cognitive Reasoning Framework")

scene_model, kb, detector, verifier, reasoner, output_gen = load_models()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp and process
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)
    
    with st.spinner("Analyzing Scene..."):
        # 1. Scene
        scene_type, caption = scene_model.analyze_scene(image)
        st.success(f"Scene Detected: {scene_type.upper()}")
        st.write(f"Caption: *{caption}*")
        
        # 2. Rules
        requirements = kb.get_scene_requirements(scene_type)
        if requirements.get("mandatory_ppe"):
            st.info(f"Required PPE: {', '.join(requirements['mandatory_ppe'])}")
        
        # 3. Detection
        detections = detector.detect(image)
        st.write(f"Detected Objects: {len(detections)}")
        
        # 4. Verification
        mandatory_ppe = requirements.get("mandatory_ppe", [])
        person_attributes = {}
        for det in detections:
            if det['label'] == 'person':
                pid = det['id']
                res = verifier.verify_attributes(image, det['box'], mandatory_ppe)
                person_attributes[pid] = res
                
        # 5. Reasoning
        result = reasoner.analyze_compliance(scene_type, detections, person_attributes, requirements)
        
        # 6. Output
        st.divider()
        st.subheader("Compliance Report")
        
        report = output_gen.generate_report(result)
        st.text(report)
        
        if result['overall_status'] == "COMPLIANT":
            st.success("Analysis Complete: No safety violations found.")
        else:
            st.error("Violations Detected!")

