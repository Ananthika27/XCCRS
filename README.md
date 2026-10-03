XCCRS: Vision-Language Cognitive Reasoning Framework for Automated Safety Compliance

An Open-Source Neuro-Symbolic System for Industrial Safety Monitoring

Overview

XCCRS (eXtended Cognitive Compliance Reasoning System) is a research-grade framework designed to detect safety violations in industrial environments. Unlike black-box LLM approaches, XCCRS uses a Neuro-Symbolic Architecture that separates neural perception (Vision-Language Models) from symbolic reasoning (Deterministic Logic). This ensures 100% explainability, reproducibility, and reliability—critical for safety-critical applications.

Key Features:

Zero-Shot Scene Understanding using BLIP.
Object & Hazard Detection using YOLOv8.
Fine-Grained Attribute Verification using CLIP.
Deterministic Neuro-Symbolic Logic for compliance checking.
Fully Open Source and free (No API keys required).
Architecture

[INPUT IMAGE]
      │
      ├───> [SCENE UNDERSTANDING] (BLIP) ──────┐
      │     (Generate Caption -> Extract Type) │
      │                                        ▼
      ├───> [OBJECT DETECTION] (YOLOv8) ──> [REASONING ENGINE] <── [KNOWLEDGE BASE]
      │     (People, Hazards)                  ▲       │             (Safety Rules JSON)
      │            │                           │       │
      │            ▼                           │       │
      └───> [ATTRIBUTE VERIFICATION] (CLIP) ───┘       │
            (Verify PPE: Helmet, Vest...)              │
                                                       ▼
                                            [OUTPUT GENERATOR]
                                            (Report, Alerts)
System Requirements

Python 3.10+
8GB RAM minimum (16GB recommended for smooth CLIP/BLIP execution)
GPU recommended (CUDA or MPS) but runs on CPU.
Installation

Navigate to the project directory:

cd Desktop/XCCRS
Activate the virtual environment:

source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Quick Start

Command Line Interface: Run the main pipeline on an image or video:

python xccrs/main.py --input path/to/factory_image.jpg
Interactive Demo: Launch the Streamlit dashboard:

streamlit run xccrs/demo_app.py
Examples: Run the example script to see the reasoning engine in action:

python xccrs/examples_quick_start.py
Project Structure

modules/: Core logic separated by cognitive function.
configs/: JSON-based safety rules (editable).
main.py: CLI entry point.
demo_app.py: Web-based GUI.
License

MIT License. Free for research and educational use.
