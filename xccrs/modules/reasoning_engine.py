import logging
import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    NEURO-SYMBOLIC COGNITIVE REASONING ENGINE.
    This module applies deterministic logical rules to perceptual inputs.
    It is the 'Symbolic' part of the system.
    """
    
    def __init__(self):
        pass

    def analyze_compliance(self, 
                           scene_type: str, 
                           detections: List[Dict[str, Any]], 
                           person_attributes: Dict[int, Dict[str, bool]],
                           requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates safety rules against the current scene state.
        
        Args:
            scene_type: Detected scene (e.g., 'factory').
            detections: List of YOLO detections.
            person_attributes: Dict mapping detection_id to verified attributes (from CLIP).
            requirements: Loading rules from KnowledgeBase for this scene.
            
        Returns:
            Dict containing violation status, trace, severity, and recommendations.
        """
        
        trace = []
        violations = []
        recommendations = set()
        
        trace.append(f"Starting safety analysis for {scene_type} environment")
        
        if not requirements or not requirements.get("rules"):
            trace.append("No specific rules found for this scene type.")
            return {
                "overall_status": "UNKNOWN",
                "violations": [],
                "reasoning_trace": trace,
                "severity": "LOW",
                "recommendations": []
            }

        # Context collection
        # We need to abstract the detections into a symbolic state
        # e.g. state = {"person_present": True, "fire_detected": False, ...}
        
        people_ids = [d['id'] for d in detections if d['label'] == 'person']
        people_count = len(people_ids)
        trace.append(f"Detected {people_count} person(s) in scene")
        
        mandatory_ppe = requirements.get("mandatory_ppe", [])
        trace.append(f"Required PPE for {scene_type}: {mandatory_ppe}")

        # 1. Check Global Hazards (from detections)
        danger_signals = requirements.get("danger_signals", [])
        for det in detections:
            label = det['label']
            if label in danger_signals: # e.g. 'fire'
                violations.append({
                    "rule": "ENV-HAZARD",
                    "severity": "CRITICAL",
                    "description": f"Dangerous hazard detected: {label}"
                })
                trace.append(f"CRITICAL: Detected danger signal '{label}'")
                recommendations.add(f"Evacuate area immediately due to {label}")

        # 2. Check PPE Compliance for each Person
        for pid in people_ids:
            # We map the internal YOLO detection index to the person attributes
            attrs = person_attributes.get(pid, {})
            
            missing_ppe = []
            for ppe in mandatory_ppe:
                # If mapped attribute is False, it's missing
                # Note: We rely on CLIP having verified these keys. 
                # If CLIP wasn't asked, we assume missing or handle gracefully.
                # Here we assume AttributeVerifier was called for all mandatory_ppe.
                if not attrs.get(ppe, False):
                    missing_ppe.append(ppe)
            
            if missing_ppe:
                # Violation!
                v_desc = f"Person (id={pid}) missing PPE: {', '.join(missing_ppe)}"
                violations.append({
                    "rule": "PPE-VIOLATION",
                    "severity": "HIGH",
                    "description": v_desc,
                    "missing": missing_ppe
                })
                trace.append(f"VIOLATION [Person {pid}]: {v_desc}")
                for m in missing_ppe:
                    recommendations.add(f"Provide {m} to personnel")
            else:
                trace.append(f"Person {pid} is compliant with PPE.")

        # 3. Symbolic Rule Engine (Explicit Rules from JSON)
        # This handles more complex conditions like "condition": "person_near_machine"
        # Since we don't have spatial logic in this simple JSON structure, we infer context.
        # But we can check simplified conditions if we had them implemented as functions.
        # For this prototype, we map specific rule IDs to logic checks.
        
        scene_rules = requirements.get("rules", [])
        for rule in scene_rules:
            rid = rule['id']
            # Example: PPE-001 (helmet_and_vest_required)
            # Already covered by generic PPE loop above, but let's see if there are others.
            # Example: ENV-001 (fire -> critical)
            # Covered strictly by hazard loop.
            
            # This section serves to demonstrate extensibility. 
            # Real neuro-symbolic systems would traverse a graph here.
            pass

        # Determine overall status
        if any(v['severity'] == 'CRITICAL' for v in violations):
            overall_status = "NON_COMPLIANT_CRITICAL"
            overall_severity = "CRITICAL"
        elif any(v['severity'] == 'HIGH' for v in violations):
            overall_status = "NON_COMPLIANT_HIGH"
            overall_severity = "HIGH"
        elif violations:
            overall_status = "NON_COMPLIANT_MEDIUM"
            overall_severity = "MEDIUM"
        else:
            overall_status = "COMPLIANT"
            overall_severity = "SAFE"
            trace.append("No violations detected.")

        trace.append(f"Analysis complete: {len(violations)} violation(s) found")

        return {
            "overall_status": overall_status,
            "violations": violations,
            "reasoning_trace": trace,
            "severity": overall_severity,
            "recommendations": list(recommendations)
        }
