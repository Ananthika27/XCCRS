import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OutputGenerator:
    """
    Generates human-readable reports from the reasoning engine's symbolic output.
    Ensures explainability by converting the reasoning trace into text.
    """
    
    def generate_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        Creates a formatted text report.
        """
        status = analysis_result.get("overall_status", "UNKNOWN")
        severity = analysis_result.get("severity", "UNKNOWN")
        violations = analysis_result.get("violations", [])
        recommendations = analysis_result.get("recommendations", [])
        trace = analysis_result.get("reasoning_trace", [])
        
        lines = []
        lines.append("SAFETY COMPLIANCE ANALYSIS REPORT")
        lines.append("="*80)
        lines.append(f"Overall Status: {status}")
        lines.append(f"Severity Level: {severity}")
        lines.append("")
        
        if violations:
            lines.append(f"VIOLATIONS DETECTED ({len(violations)})")
            for idx, viol in enumerate(violations, 1):
                lines.append(f"{idx}. {viol.get('rule', 'VIOLATION')} ({viol.get('severity', 'UNKNOWN')})")
                lines.append(f"   Description: {viol.get('description', '')}")
                if 'missing' in viol:
                    lines.append(f"   Missing PPE: {', '.join(viol['missing'])}")
                lines.append("")
        else:
            lines.append("NO VIOLATIONS DETECTED.")
            lines.append("Environment appears compliant with safety standards.")
            lines.append("")
            
        if recommendations:
            lines.append("RECOMMENDATIONS")
            for idx, rec in enumerate(recommendations, 1):
                lines.append(f"{idx}. {rec}")
            lines.append("")
            
        lines.append("DETAILED REASONING TRACE")
        for step in trace:
            lines.append(f"- {step}")
            
        lines.append("="*80)
        
        report = "\n".join(lines)
        return report

class ActionSimulator:
    """
    Simulates industrial responses (alerts, assignments, logs).
    In a real system, this would interface with PLCs or SCADA.
    """
    
    def execute_actions(self, analysis_result: Dict[str, Any]):
        """
        Simulate actions based on severity.
        """
        severity = analysis_result.get("severity", "LOW")
        
        if severity == "CRITICAL":
            logger.critical(">>> ACTION: INITIATING EMERGENCY SHUTDOWN Protocol")
            logger.critical(">>> ACTION: Triggering Global Alarm")
            logger.critical(">>> ACTION: Notifying Safety Manager immediately")
            
        elif severity == "HIGH":
            logger.warning(">>> ACTION: Triggering Local Alarm")
            logger.warning(">>> ACTION: Logging incident for immediate review")
            logger.warning(">>> ACTION: Sending alert to Floor Supervisor")
            
        elif severity == "MEDIUM":
            logger.info(">>> ACTION: Logging warning event")
            logger.info(">>> ACTION: Displaying safety reminder on HUD")
            
        else:
            logger.info(">>> ACTION: Log entry - Routine Check Scanned (Clear)")

if __name__ == "__main__":
    pass
