"""
AI Explainability Table Generator

Creates structured explainability tables for clinical trial eligibility decisions.
Follows FDA/EMA guidance for AI transparency in medical decisions.

Table Format:
| Criterion | Patient Data | Match Status | Confidence | Evidence | Reasoning |
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ExplainabilityRow:
    """Single row in the explainability table."""
    criterion_id: str
    criterion_type: str  # inclusion or exclusion
    criterion_text: str
    patient_data_field: str
    patient_value: Any
    match_status: str  # MATCH, NO_MATCH, UNCERTAIN, MISSING_DATA
    confidence: float
    evidence_sources: List[str]
    reasoning_steps: List[str]
    concerns: List[str]


class ExplainabilityTableGenerator:
    """
    Generates AI explainability tables for screening decisions.

    The table provides criterion-by-criterion breakdown of:
    - What criterion was evaluated
    - What patient data was used
    - What the matching result was
    - How confident the system is
    - What evidence supports the decision
    - The reasoning process

    This supports:
    1. Clinical reviewer understanding
    2. Regulatory compliance (FDA/EMA)
    3. Audit trail requirements
    4. Patient safety verification
    """

    def __init__(self):
        self.rows: List[ExplainabilityRow] = []

    def add_row(
        self,
        criterion_id: str,
        criterion_type: str,
        criterion_text: str,
        patient_data_field: str,
        patient_value: Any,
        match_status: str,
        confidence: float,
        evidence_sources: List[str],
        reasoning_steps: List[str],
        concerns: Optional[List[str]] = None
    ) -> None:
        """
        Add a row to the explainability table.

        Args:
            criterion_id: Unique criterion identifier
            criterion_type: "inclusion" or "exclusion"
            criterion_text: Original criterion text
            patient_data_field: Name of patient data field used
            patient_value: Value from patient data
            match_status: MATCH, NO_MATCH, UNCERTAIN, or MISSING_DATA
            confidence: Confidence score (0.0 to 1.0)
            evidence_sources: List of evidence sources
            reasoning_steps: Step-by-step reasoning
            concerns: Any concerns or flags
        """
        row = ExplainabilityRow(
            criterion_id=criterion_id,
            criterion_type=criterion_type,
            criterion_text=criterion_text,
            patient_data_field=patient_data_field,
            patient_value=patient_value,
            match_status=match_status,
            confidence=confidence,
            evidence_sources=evidence_sources,
            reasoning_steps=reasoning_steps,
            concerns=concerns or []
        )
        self.rows.append(row)

    def from_matching_results(
        self,
        matching_results: List[Dict[str, Any]]
    ) -> "ExplainabilityTableGenerator":
        """
        Build table from matching results.

        Args:
            matching_results: List of matching results from EligibilityMatcher

        Returns:
            Self for chaining
        """
        for result in matching_results:
            self.add_row(
                criterion_id=result.get("criterion_id", ""),
                criterion_type=result.get("type", "inclusion"),
                criterion_text=result.get("criterion_text", ""),
                patient_data_field=result.get("patient_data_used", {}).get("field", ""),
                patient_value=result.get("patient_data_used", {}).get("value", ""),
                match_status=result.get("match_status", "UNKNOWN"),
                confidence=result.get("confidence", 0.0),
                evidence_sources=result.get("evidence", []),
                reasoning_steps=[result.get("reasoning", "")],
                concerns=result.get("concerns", [])
            )
        return self

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert table to list of dictionaries.

        Returns:
            List of row dictionaries
        """
        return [
            {
                "criterion_id": row.criterion_id,
                "criterion_type": row.criterion_type,
                "criterion_text": row.criterion_text,
                "patient_data_field": row.patient_data_field,
                "patient_value": str(row.patient_value),
                "match_status": row.match_status,
                "confidence": row.confidence,
                "evidence_sources": row.evidence_sources,
                "reasoning": " → ".join(row.reasoning_steps),
                "concerns": row.concerns
            }
            for row in self.rows
        ]

    def to_markdown(self) -> str:
        """
        Generate markdown table representation.

        Returns:
            Markdown formatted table
        """
        lines = []
        lines.append("## AI Explainability Table")
        lines.append("")
        lines.append("| ID | Type | Criterion | Patient Value | Status | Confidence |")
        lines.append("|---|---|---|---|---|---|")

        for row in self.rows:
            criterion_short = row.criterion_text[:50] + "..." if len(row.criterion_text) > 50 else row.criterion_text
            value_short = str(row.patient_value)[:20]
            confidence_pct = f"{row.confidence:.0%}"

            status_emoji = {
                "MATCH": "✅",
                "NO_MATCH": "❌",
                "UNCERTAIN": "⚠️",
                "MISSING_DATA": "❓"
            }.get(row.match_status, "❓")

            lines.append(
                f"| {row.criterion_id} | {row.criterion_type} | {criterion_short} | "
                f"{value_short} | {status_emoji} {row.match_status} | {confidence_pct} |"
            )

        return "\n".join(lines)

    def to_html(self) -> str:
        """
        Generate HTML table representation for UI display.

        Returns:
            HTML formatted table
        """
        html_parts = [
            "<table class='explainability-table'>",
            "<thead>",
            "<tr>",
            "<th>ID</th>",
            "<th>Type</th>",
            "<th>Criterion</th>",
            "<th>Patient Value</th>",
            "<th>Status</th>",
            "<th>Confidence</th>",
            "<th>Reasoning</th>",
            "</tr>",
            "</thead>",
            "<tbody>"
        ]

        status_colors = {
            "MATCH": "#28a745",
            "NO_MATCH": "#dc3545",
            "UNCERTAIN": "#ffc107",
            "MISSING_DATA": "#6c757d"
        }

        for row in self.rows:
            color = status_colors.get(row.match_status, "#6c757d")
            reasoning_html = "<br>→ ".join(row.reasoning_steps)

            html_parts.append(f"""
            <tr>
                <td>{row.criterion_id}</td>
                <td>{row.criterion_type}</td>
                <td title="{row.criterion_text}">{row.criterion_text[:60]}...</td>
                <td>{row.patient_value}</td>
                <td style="background-color: {color}; color: white;">{row.match_status}</td>
                <td>{row.confidence:.0%}</td>
                <td>{reasoning_html}</td>
            </tr>
            """)

        html_parts.extend(["</tbody>", "</table>"])

        return "\n".join(html_parts)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from the table.

        Returns:
            Dictionary with summary statistics
        """
        total = len(self.rows)
        if total == 0:
            return {"total": 0}

        status_counts = {
            "MATCH": 0,
            "NO_MATCH": 0,
            "UNCERTAIN": 0,
            "MISSING_DATA": 0
        }

        inclusion_results = []
        exclusion_results = []
        all_concerns = []
        confidences = []

        for row in self.rows:
            status_counts[row.match_status] = status_counts.get(row.match_status, 0) + 1
            confidences.append(row.confidence)
            all_concerns.extend(row.concerns)

            if row.criterion_type == "inclusion":
                inclusion_results.append(row.match_status)
            else:
                exclusion_results.append(row.match_status)

        # Determine overall eligibility
        inclusion_passed = all(s == "MATCH" for s in inclusion_results)
        exclusion_passed = all(s != "MATCH" for s in exclusion_results)  # NO_MATCH on exclusion is good
        has_uncertain = status_counts["UNCERTAIN"] > 0 or status_counts["MISSING_DATA"] > 0

        if inclusion_passed and exclusion_passed and not has_uncertain:
            eligibility = "ELIGIBLE"
        elif not inclusion_passed or not exclusion_passed:
            eligibility = "INELIGIBLE"
        else:
            eligibility = "UNCERTAIN"

        return {
            "total_criteria": total,
            "status_counts": status_counts,
            "inclusion_count": len(inclusion_results),
            "exclusion_count": len(exclusion_results),
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "total_concerns": len(all_concerns),
            "unique_concerns": list(set(all_concerns)),
            "eligibility_determination": eligibility
        }


class AuditTrail:
    """
    Maintains audit trail for regulatory compliance.

    Tracks all decisions and their supporting evidence
    for FDA/EMA audit requirements.
    """

    def __init__(self, screening_id: str):
        self.screening_id = screening_id
        self.created_at = datetime.now().isoformat()
        self.entries: List[Dict[str, Any]] = []

    def log_entry(
        self,
        action: str,
        details: Dict[str, Any],
        actor: str = "system"
    ) -> None:
        """
        Log an audit trail entry.

        Args:
            action: Action performed
            details: Action details
            actor: Who/what performed the action
        """
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "actor": actor,
            "details": details
        })

    def to_json(self) -> str:
        """Export audit trail as JSON."""
        return json.dumps({
            "screening_id": self.screening_id,
            "created_at": self.created_at,
            "entries": self.entries
        }, indent=2)
