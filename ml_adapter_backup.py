"""
Adapter for the existing Military Drug Kit Optimizer ML prediction logic.

The adapter keeps the trained model/prediction implementation in src/ intact.
It tries to use an existing callable when the project exposes one, while
providing a safe, explicit result when the existing module needs UI-specific
state that cannot be reconstructed by the API layer.

No model is retrained or replaced here.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional


CANDIDATE_FUNCTIONS = (
    "predict_drug",
    "predict",
    "make_prediction",
    "predict_drug_recommendation",
    "get_prediction",
)


def _find_predictor(module: Any) -> Optional[Callable[..., Any]]:
    for name in CANDIDATE_FUNCTIONS:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def _call_predictor(fn: Callable[..., Any], inputs: Dict[str, Any]) -> Any:
    """Call an existing predictor using the smallest compatible input form."""
    try:
        return fn(inputs)
    except TypeError:
        pass

    try:
        return fn(**inputs)
    except TypeError:
        pass

    signature = inspect.signature(fn)
    kwargs = {}
    aliases = {
        "symptom": ["symptom", "symptoms"],
        "condition": ["condition", "use_case", "usecase"],
        "region": ["region"],
        "mission_type": ["mission_type", "mission"],
        "environment": ["environment"],
    }

    for parameter in signature.parameters.values():
        for source_key, possible_names in aliases.items():
            if parameter.name in possible_names and source_key in inputs:
                kwargs[parameter.name] = inputs[source_key]
                break

    if kwargs:
        return fn(**kwargs)

    raise TypeError(
        f"Existing predictor {getattr(fn, '__name__', fn)!r} could not be "
        "called with the available API inputs."
    )


def predict_with_existing_model(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the existing src.predict predictor if it exposes a compatible
    callable. The original model code is never modified.
    """
    try:
        module = importlib.import_module("src.predict")
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "prediction": None,
            "source": "existing_ml_model",
            "message": f"Could not import src.predict: {exc}",
        }

    predictor = _find_predictor(module)

    if predictor is None:
        return {
            "status": "UNAVAILABLE",
            "prediction": None,
            "source": "existing_ml_model",
            "message": (
                "src.predict does not expose a reusable prediction function. "
                "Existing ML code was intentionally not rewritten. "
                "The existing Streamlit prediction path remains unchanged."
            ),
        }

    try:
        result = _call_predictor(predictor, inputs)
        return {
            "status": "SUCCESS",
            "prediction": result,
            "source": "existing_ml_model",
            "predictor": predictor.__name__,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "prediction": None,
            "source": "existing_ml_model",
            "predictor": getattr(predictor, "__name__", "unknown"),
            "message": f"Existing predictor call failed: {exc}",
        }
