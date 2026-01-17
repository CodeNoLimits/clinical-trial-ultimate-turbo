"""
Confidence Scoring Module

Implements self-consistency based confidence scoring for LLM outputs.
Uses multiple independent generations to estimate decision reliability.

Key concepts:
- Self-consistency: Agreement rate across multiple generations
- Epistemic uncertainty: Model knowledge gaps
- Aleatoric uncertainty: Data variability
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""
    overall_confidence: float
    confidence_level: str  # HIGH, MODERATE, LOW, VERY_LOW
    individual_scores: List[Dict[str, float]]
    consistency_score: float
    uncertainty_type: str  # epistemic, aleatoric, or mixed
    requires_human_review: bool
    review_reasons: List[str]


class ConfidenceScorer:
    """
    Calculates confidence scores for eligibility decisions.

    Uses a combination of:
    1. Individual criterion confidence aggregation
    2. Self-consistency measurement
    3. Missing data penalties
    4. Uncertainty classification

    Confidence Levels:
    - HIGH (≥90%): Proceed with confidence
    - MODERATE (80-89%): Proceed with monitoring
    - LOW (70-79%): Recommend human review
    - VERY_LOW (<70%): Require human review
    """

    # Confidence thresholds
    THRESHOLD_HIGH = 0.90
    THRESHOLD_MODERATE = 0.80
    THRESHOLD_LOW = 0.70

    # Penalty weights
    MISSING_DATA_PENALTY = 0.05
    UNCERTAIN_PENALTY = 0.10
    CRITICAL_CRITERION_WEIGHT = 1.5

    def __init__(
        self,
        n_samples: int = 5,
        human_review_threshold: float = 0.80
    ):
        """
        Initialize the confidence scorer.

        Args:
            n_samples: Number of samples for self-consistency
            human_review_threshold: Threshold below which human review is required
        """
        self.n_samples = n_samples
        self.human_review_threshold = human_review_threshold

    def calculate_confidence(
        self,
        matching_results: List[Dict[str, Any]],
        decision_samples: List[str] = None
    ) -> ConfidenceResult:
        """
        Calculate overall confidence for a screening decision.

        Args:
            matching_results: List of criterion matching results
            decision_samples: Optional list of decisions from multiple generations

        Returns:
            ConfidenceResult with detailed confidence information
        """
        # Step 1: Extract individual confidences
        individual_scores = []
        for result in matching_results:
            individual_scores.append({
                "criterion_id": result.get("criterion_id", ""),
                "confidence": result.get("confidence", 0.5),
                "status": result.get("match_status", "UNKNOWN"),
                "is_critical": result.get("is_critical", False)
            })

        # Step 2: Calculate base confidence (weighted average)
        base_confidence = self._calculate_weighted_average(individual_scores)

        # Step 3: Apply penalties
        penalties, penalty_reasons = self._calculate_penalties(matching_results)
        penalized_confidence = max(0.0, base_confidence - penalties)

        # Step 4: Calculate self-consistency if samples provided
        if decision_samples and len(decision_samples) > 1:
            consistency_score = self._calculate_consistency(decision_samples)
            # Adjust confidence based on consistency
            final_confidence = (penalized_confidence * 0.7) + (consistency_score * 0.3)
        else:
            consistency_score = 1.0  # Assume perfect consistency if no samples
            final_confidence = penalized_confidence

        # Step 5: Classify confidence level
        confidence_level = self._classify_confidence(final_confidence)

        # Step 6: Determine uncertainty type
        uncertainty_type = self._classify_uncertainty(matching_results, consistency_score)

        # Step 7: Determine if human review needed
        requires_review = final_confidence < self.human_review_threshold
        review_reasons = []

        if requires_review:
            review_reasons.extend(penalty_reasons)
            if consistency_score < 0.8:
                review_reasons.append("Low consistency across assessments")
            if confidence_level in ["LOW", "VERY_LOW"]:
                review_reasons.append(f"Confidence level is {confidence_level}")

        return ConfidenceResult(
            overall_confidence=final_confidence,
            confidence_level=confidence_level,
            individual_scores=individual_scores,
            consistency_score=consistency_score,
            uncertainty_type=uncertainty_type,
            requires_human_review=requires_review,
            review_reasons=review_reasons
        )

    def _calculate_weighted_average(
        self,
        individual_scores: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate weighted average of individual confidences.

        Critical criteria get higher weight.
        """
        if not individual_scores:
            return 0.5

        total_weight = 0.0
        weighted_sum = 0.0

        for score in individual_scores:
            weight = self.CRITICAL_CRITERION_WEIGHT if score.get("is_critical") else 1.0
            weighted_sum += score["confidence"] * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _calculate_penalties(
        self,
        matching_results: List[Dict[str, Any]]
    ) -> Tuple[float, List[str]]:
        """
        Calculate penalties based on uncertain/missing data.

        Returns tuple of (total_penalty, reasons).
        """
        total_penalty = 0.0
        reasons = []

        missing_count = 0
        uncertain_count = 0

        for result in matching_results:
            status = result.get("match_status", "")
            is_critical = result.get("is_critical", False)

            if status == "MISSING_DATA":
                missing_count += 1
                penalty = self.MISSING_DATA_PENALTY
                if is_critical:
                    penalty *= 2
                    reasons.append(f"Missing critical data: {result.get('criterion_id', 'unknown')}")
                total_penalty += penalty

            elif status == "UNCERTAIN":
                uncertain_count += 1
                penalty = self.UNCERTAIN_PENALTY
                if is_critical:
                    penalty *= 2
                    reasons.append(f"Uncertain critical criterion: {result.get('criterion_id', 'unknown')}")
                total_penalty += penalty

        if missing_count > 0:
            reasons.append(f"{missing_count} criteria have missing data")
        if uncertain_count > 0:
            reasons.append(f"{uncertain_count} criteria are uncertain")

        return total_penalty, reasons

    def _calculate_consistency(
        self,
        decision_samples: List[str]
    ) -> float:
        """
        Calculate self-consistency score from multiple decision samples.

        Higher score = more agreement across samples.
        """
        if not decision_samples:
            return 1.0

        # Count occurrences of each decision
        decision_counts = {}
        for decision in decision_samples:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        # Get most common decision
        max_count = max(decision_counts.values())

        # Consistency = proportion of samples agreeing with majority
        return max_count / len(decision_samples)

    def _classify_confidence(self, confidence: float) -> str:
        """Classify confidence into level."""
        if confidence >= self.THRESHOLD_HIGH:
            return "HIGH"
        elif confidence >= self.THRESHOLD_MODERATE:
            return "MODERATE"
        elif confidence >= self.THRESHOLD_LOW:
            return "LOW"
        else:
            return "VERY_LOW"

    def _classify_uncertainty(
        self,
        matching_results: List[Dict[str, Any]],
        consistency_score: float
    ) -> str:
        """
        Classify uncertainty type.

        - Epistemic: Model knowledge gaps (can be reduced with more data)
        - Aleatoric: Inherent data variability (cannot be reduced)
        - Mixed: Both types present
        """
        missing_count = sum(
            1 for r in matching_results
            if r.get("match_status") == "MISSING_DATA"
        )

        uncertain_count = sum(
            1 for r in matching_results
            if r.get("match_status") == "UNCERTAIN"
        )

        # Epistemic: missing data or low consistency suggests knowledge gap
        has_epistemic = missing_count > 0 or consistency_score < 0.8

        # Aleatoric: uncertain matches with high consistency suggests data variability
        has_aleatoric = uncertain_count > 0 and consistency_score >= 0.8

        if has_epistemic and has_aleatoric:
            return "mixed"
        elif has_epistemic:
            return "epistemic"
        elif has_aleatoric:
            return "aleatoric"
        else:
            return "none"

    def score_to_dict(self, result: ConfidenceResult) -> Dict[str, Any]:
        """Convert ConfidenceResult to dictionary for serialization."""
        return {
            "overall_confidence": result.overall_confidence,
            "confidence_level": result.confidence_level,
            "individual_scores": result.individual_scores,
            "consistency_score": result.consistency_score,
            "uncertainty_type": result.uncertainty_type,
            "requires_human_review": result.requires_human_review,
            "review_reasons": result.review_reasons
        }
