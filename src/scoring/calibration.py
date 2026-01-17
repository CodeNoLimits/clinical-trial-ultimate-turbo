"""
Probability Calibration Module

Implements calibration techniques to ensure confidence scores
accurately reflect true probabilities of correctness.

Key techniques:
- Temperature scaling
- Isotonic regression
- Platt scaling
"""

from typing import List, Tuple, Optional
import numpy as np


class ProbabilityCalibrator:
    """
    Calibrates probability outputs from LLMs.

    LLMs often produce poorly calibrated confidence scores.
    This module adjusts scores to better reflect true accuracy.

    Methods:
    - Temperature scaling: Simple single-parameter scaling
    - Isotonic regression: Non-parametric monotonic calibration
    - Platt scaling: Sigmoid-based calibration
    """

    def __init__(self, method: str = "temperature"):
        """
        Initialize the calibrator.

        Args:
            method: Calibration method - "temperature", "isotonic", or "platt"
        """
        self.method = method
        self.temperature = 1.0
        self.isotonic_model = None
        self.platt_params = None
        self._is_fitted = False

    def fit(
        self,
        predictions: List[float],
        actuals: List[bool],
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrator on validation data.

        Args:
            predictions: Predicted probabilities (0-1)
            actuals: Actual outcomes (True/False)

        Returns:
            Self for chaining
        """
        predictions = np.array(predictions)
        actuals = np.array(actuals).astype(float)

        if self.method == "temperature":
            self.temperature = self._fit_temperature(predictions, actuals)

        elif self.method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
                self.isotonic_model = IsotonicRegression(out_of_bounds="clip")
                self.isotonic_model.fit(predictions, actuals)
            except ImportError:
                print("Warning: sklearn not available, falling back to temperature scaling")
                self.method = "temperature"
                self.temperature = self._fit_temperature(predictions, actuals)

        elif self.method == "platt":
            self.platt_params = self._fit_platt(predictions, actuals)

        self._is_fitted = True
        return self

    def calibrate(self, probability: float) -> float:
        """
        Calibrate a single probability.

        Args:
            probability: Raw probability (0-1)

        Returns:
            Calibrated probability (0-1)
        """
        if not self._is_fitted:
            return probability  # Return unchanged if not fitted

        if self.method == "temperature":
            # Temperature scaling
            logit = np.log(probability / (1 - probability + 1e-10))
            scaled_logit = logit / self.temperature
            return 1 / (1 + np.exp(-scaled_logit))

        elif self.method == "isotonic" and self.isotonic_model:
            return float(self.isotonic_model.predict([probability])[0])

        elif self.method == "platt" and self.platt_params:
            a, b = self.platt_params
            return 1 / (1 + np.exp(a * probability + b))

        return probability

    def calibrate_batch(self, probabilities: List[float]) -> List[float]:
        """
        Calibrate a batch of probabilities.

        Args:
            probabilities: List of raw probabilities

        Returns:
            List of calibrated probabilities
        """
        return [self.calibrate(p) for p in probabilities]

    def _fit_temperature(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        Fit temperature parameter using NLL optimization.

        Uses grid search for simplicity.
        """
        best_temp = 1.0
        best_nll = float("inf")

        for temp in np.linspace(0.5, 3.0, 50):
            # Apply temperature scaling
            logits = np.log(predictions / (1 - predictions + 1e-10))
            scaled = 1 / (1 + np.exp(-logits / temp))

            # Calculate NLL
            nll = -np.mean(
                actuals * np.log(scaled + 1e-10) +
                (1 - actuals) * np.log(1 - scaled + 1e-10)
            )

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        return best_temp

    def _fit_platt(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit Platt scaling parameters (a, b).

        Uses gradient descent for simplicity.
        """
        # Initialize parameters
        a, b = 0.0, 0.0
        lr = 0.1

        for _ in range(100):
            # Forward pass
            sigmoid = 1 / (1 + np.exp(a * predictions + b))

            # Compute gradients
            error = sigmoid - actuals
            grad_a = np.mean(error * predictions)
            grad_b = np.mean(error)

            # Update parameters
            a -= lr * grad_a
            b -= lr * grad_b

        return (a, b)

    def compute_ece(
        self,
        predictions: List[float],
        actuals: List[bool],
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.

        ECE measures the difference between predicted confidence
        and actual accuracy across bins.

        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes
            n_bins: Number of bins for ECE calculation

        Returns:
            ECE score (lower is better, 0 is perfectly calibrated)
        """
        predictions = np.array(predictions)
        actuals = np.array(actuals).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (predictions > bin_edges[i]) & (predictions <= bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(actuals[mask])
                bin_conf = np.mean(predictions[mask])
                bin_size = np.sum(mask) / len(predictions)
                ece += bin_size * np.abs(bin_acc - bin_conf)

        return float(ece)

    def reliability_diagram_data(
        self,
        predictions: List[float],
        actuals: List[bool],
        n_bins: int = 10
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Generate data for reliability diagram.

        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes
            n_bins: Number of bins

        Returns:
            Tuple of (mean_predictions, accuracies, counts) per bin
        """
        predictions = np.array(predictions)
        actuals = np.array(actuals).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        mean_preds = []
        accuracies = []
        counts = []

        for i in range(n_bins):
            mask = (predictions > bin_edges[i]) & (predictions <= bin_edges[i + 1])
            count = np.sum(mask)

            if count > 0:
                mean_preds.append(float(np.mean(predictions[mask])))
                accuracies.append(float(np.mean(actuals[mask])))
                counts.append(int(count))
            else:
                mean_preds.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                accuracies.append(0.0)
                counts.append(0)

        return mean_preds, accuracies, counts
