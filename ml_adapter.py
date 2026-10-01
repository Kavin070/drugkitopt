"""
Adapter between the Agentic AI workflow and the existing ML predictor.

Important:
- Does NOT modify model.pkl.
- Does NOT retrain the model.
- Only converts user-friendly inputs into categories
  already known by the existing model.
"""

from typing import Any, Dict, List, Optional
import difflib


# ---------------------------------------------------------------------
# Friendly input -> exact dataset/model categories
# ---------------------------------------------------------------------

REGION_MAP = {
    "tropical": "Tropical rainforest; malaria, parasites, fungal skin disease, high humidity",
    "tropical rainforest": "Tropical rainforest; malaria, parasites, fungal skin disease, high humidity",
    "jungle": "Tropical rainforest; malaria, parasites, fungal skin disease, high humidity",

    "desert": "Arid heat; dehydration, heat illness, dust exposure",
    "arid": "Arid heat; dehydration, heat illness, dust exposure",

    "mountain": "Mountain; AMS/HAPE/HACE risk, cold",
    "high altitude": "Mountain; AMS/HAPE/HACE risk, cold",
    "altitude": "Mountain; AMS/HAPE/HACE risk, cold",

    "urban": "Dense population; trauma, respiratory infections",
    "city": "Dense population; trauma, respiratory infections",

    "arctic": "Cold; frostbite risk, dry air",
    "cold": "Cold; frostbite risk, dry air",

    "naval": "Sea; motion sickness, vitamin D deficiency",
    "sea": "Sea; motion sickness, vitamin D deficiency",

    "peacekeeping": "Mixed environment; sanitation, chronic care",
    "camp": "Mixed environment; sanitation, chronic care",
    "mixed": "Mixed environment; sanitation, chronic care",
}


MISSION_MAP = {
    "jungle": "Jungle Ops",
    "jungle ops": "Jungle Ops",

    "desert": "Desert Ops",
    "desert ops": "Desert Ops",

    "high altitude": "High-Altitude Ops",
    "altitude": "High-Altitude Ops",
    "high-altitude": "High-Altitude Ops",

    "urban": "Urban Ops",
    "urban ops": "Urban Ops",

    "arctic": "Arctic Ops",
    "arctic ops": "Arctic Ops",

    "naval": "Naval Ops",
    "naval ops": "Naval Ops",

    "peacekeeping": "Peacekeeping/Camp Ops",
    "peacekeeping ops": "Peacekeeping/Camp Ops",
    "camp": "Peacekeeping/Camp Ops",
    "camp ops": "Peacekeeping/Camp Ops",
}


# ---------------------------------------------------------------------
# Use-case keyword matching
# ---------------------------------------------------------------------

USE_CASE_KEYWORDS = {
    "fever": [
        "Pain, fever",
    ],
    "pain": [
        "Pain, fever",
        "Acute pain (short-term)",
        "Moderate to severe pain",
        "Inflammatory pain",
    ],
    "acute pain": [
        "Acute pain (short-term)",
    ],
    "musculoskeletal": [
        "Musculoskeletal pain, inflammation",
    ],
    "inflammation": [
        "Inflammatory pain",
    ],
    "severe pain": [
        "Severe battlefield pain",
        "Severe pain (combat)",
        "Moderate to severe pain",
    ],
    "respiratory infection": [
        "Respiratory/skin infections",
    ],
    "skin infection": [
        "Respiratory/skin infections",
    ],
    "polymicrobial infection": [
        "Polymicrobial infections",
    ],
    "fungal infection": [
        "Fungal infections",
    ],
    "fungal infections": [
        "Fungal infections",
    ],
    "dermatophyte": [
        "Dermatophyte infections",
    ],
    "allergy": [
        "Allergic symptoms",
        "Allergic rhinitis/urticaria",
    ],
    "allergic": [
        "Allergic symptoms",
    ],
    "asthma": [
        "Asthma control",
        "Allergic rhinitis/asthma adjunct",
    ],
    "bronchospasm": [
        "Bronchospasm",
        "Bronchospasm (adjunct)",
    ],
    "anaphylaxis": [
        "Anaphylaxis",
    ],
    "burn": [
        "Burns",
    ],
    "burns": [
        "Burns",
    ],
    "dehydration": [
        "Dehydration",
    ],
    "insect bite": [
        "Dermatitis, insect bites",
    ],
    "insect bites": [
        "Dermatitis, insect bites",
    ],
    "diarrhea": [
        "Diarrhea adjunct (peds)/wound",
    ],
    "altitude sickness": [
        "Altitude sickness (AMS)",
    ],
    "ams": [
        "Altitude sickness (AMS)",
    ],
    "anxiety": [
        "Anxiety, itching",
    ],
    "itching": [
        "Anxiety, itching",
    ],
}


def _clean(value: Optional[str]) -> str:
    return " ".join((value or "").strip().lower().split())


def map_region(region: str) -> Optional[str]:
    """Map friendly region text to an exact trained category."""
    text = _clean(region)

    if text in REGION_MAP:
        return REGION_MAP[text]

    for key, value in REGION_MAP.items():
        if key in text:
            return value

    return None


def map_mission(mission: str) -> Optional[str]:
    """Map friendly mission text to an exact trained category."""
    text = _clean(mission)

    if text in MISSION_MAP:
        return MISSION_MAP[text]

    for key, value in MISSION_MAP.items():
        if key in text:
            return value

    return None


def map_use_case(
    symptom: str,
    condition: str,
    available_use_cases: List[str],
) -> Optional[str]:
    """
    Map user text to an existing ML Use_Case category.

    This function NEVER invents a new category. It only returns a value
    that exists in the trained encoder.
    """
    text = _clean(f"{symptom} {condition}")

    if not text:
        return None

    # First use explicit keyword mappings.
    for keyword, candidates in USE_CASE_KEYWORDS.items():
        if keyword in text:
            for candidate in candidates:
                if candidate in available_use_cases:
                    return candidate

    # Then try exact case-insensitive match.
    lowered = {
        _clean(value): value
        for value in available_use_cases
    }

    if text in lowered:
        return lowered[text]

    # Finally use conservative fuzzy matching.
    matches = difflib.get_close_matches(
        text,
        [_clean(value) for value in available_use_cases],
        n=1,
        cutoff=0.85,
    )

    if matches:
        return lowered[matches[0]]

    return None


# ---------------------------------------------------------------------
# Existing ML integration
# ---------------------------------------------------------------------

def predict_with_existing_model(
    symptom: str = "",
    condition: str = "",
    region: str = "",
    mission_type: str = "",
) -> Dict[str, Any]:
    """
    Call the existing ML prediction engine without modifying it.
    """

    try:
        from src.predict import load_model, predict_drug

        _, encoders, _ = load_model()

        available_use_cases = list(
            encoders["Use_Case"].classes_
        )

        available_regions = list(
            encoders["Region_Notes"].classes_
        )

        available_missions = list(
            encoders["Mission_Type"].classes_
        )

        use_case = map_use_case(
            symptom,
            condition,
            available_use_cases,
        )

        mapped_region = map_region(region)

        mapped_mission = map_mission(mission_type)

        # Verify that mapped values really belong to the trained encoders.
        if mapped_region not in available_regions:
            mapped_region = None

        if mapped_mission not in available_missions:
            mapped_mission = None

        if not use_case:
            return {
                "status": "UNAVAILABLE",
                "prediction": None,
                "source": "existing_ml_model",
                "message": (
                    "The supplied symptom/condition could not be mapped "
                    "confidently to an existing trained Use_Case category."
                ),
                "mapped_inputs": {
                    "use_case": None,
                    "region": mapped_region,
                    "mission": mapped_mission,
                },
            }

        if not mapped_region or not mapped_mission:
            return {
                "status": "UNAVAILABLE",
                "prediction": None,
                "source": "existing_ml_model",
                "message": (
                    "Region or mission could not be mapped to an exact "
                    "category used by the existing ML model."
                ),
                "mapped_inputs": {
                    "use_case": use_case,
                    "region": mapped_region,
                    "mission": mapped_mission,
                },
            }

        prediction = predict_drug(
            use_case,
            mapped_region,
            mapped_mission,
        )

        return {
            "status": "SUCCESS",
            "prediction": prediction,
            "source": "existing_ml_model",
            "predictor": "predict_drug",
            "mapped_inputs": {
                "use_case": use_case,
                "region": mapped_region,
                "mission": mapped_mission,
            },
        }

    except Exception as exc:
        return {
            "status": "ERROR",
            "prediction": None,
            "source": "existing_ml_model",
            "message": f"Existing predictor call failed: {exc}",
        }


# Backward-compatible name used by the Coordinator.
def run_existing_ml(
    symptom: str = "",
    condition: str = "",
    region: str = "",
    mission_type: str = "",
    **kwargs,
) -> Dict[str, Any]:
    return predict_with_existing_model(
        symptom=symptom,
        condition=condition,
        region=region,
        mission_type=mission_type,
    )