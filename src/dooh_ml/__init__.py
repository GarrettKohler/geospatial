"""DOOH Site Optimization ML Package.

Three-model architecture for site prioritization:
- Similarity: Gower distance lookalike modeling
- Causal: Double ML for hardware treatment effects
- Classifier: CatBoost for activation success prediction
"""

__version__ = "0.1.0"

from dooh_ml.models.similarity import SimilarityModel
from dooh_ml.models.causal import CausalModel
from dooh_ml.models.classifier import ActivationClassifier
from dooh_ml.inference.prioritizer import SitePrioritizer

__all__ = [
    "SimilarityModel",
    "CausalModel",
    "ActivationClassifier",
    "SitePrioritizer",
]
