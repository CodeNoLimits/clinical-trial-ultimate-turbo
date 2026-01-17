"""
Clinical Narrative Generator

Generates clinician-friendly narrative explanations of eligibility decisions.
Uses LLM to transform structured data into natural language suitable for
clinical documentation and patient communication.
"""

from typing import Dict, Any, List, Optional
import os

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from langchain_core.messages import SystemMessage, HumanMessage


class NarrativeGenerator:
    """
    Generates clinical narratives for eligibility decisions.

    Produces documentation-ready text that:
    - Summarizes the screening decision
    - Explains key factors
    - Highlights concerns
    - Provides actionable recommendations
    """

    NARRATIVE_PROMPT = """
You are a clinical documentation specialist. Generate a clear, professional
clinical narrative summarizing a patient's eligibility for a clinical trial.

The narrative should:
1. State the eligibility decision clearly
2. Summarize the key factors that led to the decision
3. Note any concerns or areas of uncertainty
4. Provide recommendations for next steps

Use appropriate medical terminology but keep it accessible.
Do NOT include patient identifying information.
Format as a single cohesive paragraph suitable for medical records.
"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the narrative generator.

        Args:
            model_name: Optional LLM model override
        """
        provider = os.getenv("LLM_PROVIDER", "google")

        if provider == "google" and ChatGoogleGenerativeAI:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name or os.getenv("LLM_MODEL", "gemini-2.0-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3,
            )
        elif provider == "openai" and ChatOpenAI:
            self.llm = ChatOpenAI(
                model=model_name or os.getenv("LLM_MODEL", "gpt-4-turbo"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3,
            )
        else:
            self.llm = None

    async def generate_narrative(
        self,
        decision: str,
        confidence: float,
        explainability_table: List[Dict[str, Any]],
        patient_profile: Dict[str, Any],
        trial_id: str
    ) -> str:
        """
        Generate a clinical narrative for the eligibility decision.

        Args:
            decision: ELIGIBLE, INELIGIBLE, or UNCERTAIN
            confidence: Confidence score (0-1)
            explainability_table: Detailed criterion results
            patient_profile: Patient demographics (anonymized)
            trial_id: Trial identifier

        Returns:
            Clinical narrative text
        """
        if self.llm is None:
            return self._generate_template_narrative(
                decision, confidence, explainability_table, trial_id
            )

        # Build context for LLM
        context = self._build_context(
            decision, confidence, explainability_table, patient_profile, trial_id
        )

        messages = [
            SystemMessage(content=self.NARRATIVE_PROMPT),
            HumanMessage(content=context)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            # Fallback to template if LLM fails
            return self._generate_template_narrative(
                decision, confidence, explainability_table, trial_id
            )

    def _build_context(
        self,
        decision: str,
        confidence: float,
        explainability_table: List[Dict[str, Any]],
        patient_profile: Dict[str, Any],
        trial_id: str
    ) -> str:
        """Build context string for LLM narrative generation."""

        # Summarize key matches and failures
        matches = [r for r in explainability_table if r.get("match_status") == "MATCH"]
        failures = [r for r in explainability_table if r.get("match_status") == "NO_MATCH"]
        uncertain = [r for r in explainability_table if r.get("match_status") in ["UNCERTAIN", "MISSING_DATA"]]

        context = f"""
ELIGIBILITY SCREENING SUMMARY

Trial: {trial_id}
Decision: {decision}
Confidence: {confidence:.0%}

Patient Demographics:
- Age: {patient_profile.get('demographics', {}).get('age', 'Unknown')} years
- Sex: {patient_profile.get('demographics', {}).get('sex', 'Unknown')}

CRITERIA EVALUATION:

Criteria Met ({len(matches)}):
"""
        for m in matches[:5]:  # Limit to top 5
            context += f"- {m.get('criterion_text', '')[:100]}\n"

        context += f"""
Criteria Not Met ({len(failures)}):
"""
        for f in failures[:5]:
            context += f"- {f.get('criterion_text', '')[:100]} (Patient: {f.get('patient_value', 'N/A')})\n"

        if uncertain:
            context += f"""
Uncertain/Missing Data ({len(uncertain)}):
"""
            for u in uncertain[:5]:
                context += f"- {u.get('criterion_text', '')[:100]}\n"

        # Add concerns
        all_concerns = []
        for row in explainability_table:
            all_concerns.extend(row.get("concerns", []))

        if all_concerns:
            context += f"""
Concerns Identified:
"""
            for concern in list(set(all_concerns))[:5]:
                context += f"- {concern}\n"

        context += """
Generate a clinical narrative summarizing this eligibility assessment.
"""

        return context

    def _generate_template_narrative(
        self,
        decision: str,
        confidence: float,
        explainability_table: List[Dict[str, Any]],
        trial_id: str
    ) -> str:
        """
        Generate narrative using templates (fallback when LLM unavailable).

        Args:
            decision: Eligibility decision
            confidence: Confidence score
            explainability_table: Criterion results
            trial_id: Trial identifier

        Returns:
            Template-based narrative
        """
        # Count statuses
        status_counts = {}
        for row in explainability_table:
            status = row.get("match_status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1

        matches = status_counts.get("MATCH", 0)
        failures = status_counts.get("NO_MATCH", 0)
        uncertain = status_counts.get("UNCERTAIN", 0) + status_counts.get("MISSING_DATA", 0)
        total = len(explainability_table)

        # Build narrative based on decision
        if decision == "ELIGIBLE":
            narrative = (
                f"The patient has been assessed as ELIGIBLE for clinical trial {trial_id} "
                f"with {confidence:.0%} confidence. "
                f"The patient met all {matches} inclusion criteria evaluated. "
            )
            if uncertain > 0:
                narrative += (
                    f"However, {uncertain} criteria could not be fully evaluated due to "
                    f"uncertain or missing data. Clinical review is recommended. "
                )

        elif decision == "INELIGIBLE":
            # Find key failure reasons
            failure_reasons = [
                row.get("criterion_text", "")[:50]
                for row in explainability_table
                if row.get("match_status") == "NO_MATCH"
            ][:3]

            narrative = (
                f"The patient has been assessed as INELIGIBLE for clinical trial {trial_id} "
                f"with {confidence:.0%} confidence. "
                f"The patient did not meet {failures} of the {total} criteria evaluated. "
            )
            if failure_reasons:
                narrative += f"Key exclusion factors include: {'; '.join(failure_reasons)}. "

        else:  # UNCERTAIN
            narrative = (
                f"The eligibility of this patient for clinical trial {trial_id} could not "
                f"be determined with sufficient confidence (current: {confidence:.0%}). "
                f"Of the {total} criteria evaluated, {uncertain} had uncertain or missing data. "
                f"Clinical review is REQUIRED before making a final determination. "
            )

        # Add concerns if any
        all_concerns = []
        for row in explainability_table:
            all_concerns.extend(row.get("concerns", []))

        if all_concerns:
            unique_concerns = list(set(all_concerns))[:3]
            narrative += f"Additional concerns noted: {'; '.join(unique_concerns)}. "

        return narrative

    def generate_recommendation(
        self,
        decision: str,
        confidence: float,
        missing_data: List[str]
    ) -> str:
        """
        Generate actionable recommendations based on screening results.

        Args:
            decision: Eligibility decision
            confidence: Confidence score
            missing_data: List of missing data fields

        Returns:
            Recommendation text
        """
        recommendations = []

        if confidence < 0.8:
            recommendations.append(
                "This assessment requires clinical review due to confidence below 80%."
            )

        if missing_data:
            recommendations.append(
                f"The following data should be obtained to improve assessment accuracy: "
                f"{', '.join(missing_data[:5])}."
            )

        if decision == "UNCERTAIN":
            recommendations.append(
                "A definitive eligibility determination could not be made. "
                "Please obtain additional clinical information and re-evaluate."
            )

        if decision == "ELIGIBLE" and confidence >= 0.9:
            recommendations.append(
                "Patient appears to be a strong candidate for enrollment. "
                "Proceed with standard enrollment procedures."
            )

        if not recommendations:
            recommendations.append(
                "Standard review procedures apply. "
                "Consult with clinical team if questions arise."
            )

        return " ".join(recommendations)
