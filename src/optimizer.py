import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(data_path="../data/dataset.csv"):
    """Load and clean the military medicine dataset."""

    df = pd.read_csv(data_path)

    df = df.drop_duplicates()

    required_columns = [
        "Drug_Name",
        "Generic_Name",
        "Use_Case",
        "Typical_Dosage",
        "Quantity_in_Kit",
        "Region_Notes",
        "Mission_Type"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df.dropna(
        subset=required_columns
    )

    return df


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """Normalize text for reliable comparison."""

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
    )


# ============================================================
# 3. REGION MATCHING
# ============================================================

def region_matches(
    requested_region,
    dataset_region
):
    """
    Check whether the medicine's regional context
    matches the requested region.
    """

    requested_region = normalize_text(
        requested_region
    )

    dataset_region = normalize_text(
        dataset_region
    )

    if not requested_region:
        return False

    if not dataset_region:
        return False

    # Exact match
    if requested_region == dataset_region:
        return True

    # Example:
    # requested = "arid heat"
    # dataset   = "arid heat; dehydration, heat illness..."
    if requested_region in dataset_region:
        return True

    return False


# ============================================================
# 4. MISSION MATCHING
# ============================================================

def mission_matches(
    requested_mission,
    dataset_mission
):
    """
    Check whether the dataset row belongs to
    the requested mission.
    """

    requested_mission = normalize_text(
        requested_mission
    )

    dataset_mission = normalize_text(
        dataset_mission
    )

    if not requested_mission:
        return False

    if not dataset_mission:
        return False

    return (
        requested_mission
        == dataset_mission
    )


# ============================================================
# 5. MEDICAL USE-CASE MATCHING
# ============================================================

def use_case_matches(
    requested_condition,
    dataset_use_case
):
    """
    IMPORTANT:
    Medical coverage is determined ONLY from Use_Case.

    Region_Notes is deliberately NOT used here.
    """

    requested_condition = normalize_text(
        requested_condition
    )

    dataset_use_case = normalize_text(
        dataset_use_case
    )

    if not requested_condition:
        return False

    if not dataset_use_case:
        return False

    # Exact match
    if requested_condition == dataset_use_case:
        return True

    # Requested phrase contained in Use_Case
    if requested_condition in dataset_use_case:
        return True

    # Dataset Use_Case contained in requested phrase
    if dataset_use_case in requested_condition:
        return True

    return False


# ============================================================
# 6. FIND CANDIDATE MEDICINES
# ============================================================

def find_candidate_medicines(
    df,
    region,
    mission,
    required_conditions
):
    """
    Find medicines that:

    1. Treat at least one required Use_Case
    2. Match the requested region
    3. Match the requested mission
    """

    candidates = []

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # Region compatibility
        # ----------------------------------------------------

        if not region_matches(
            region,
            row["Region_Notes"]
        ):
            continue

        # ----------------------------------------------------
        # Mission compatibility
        # ----------------------------------------------------

        if not mission_matches(
            mission,
            row["Mission_Type"]
        ):
            continue

        # ----------------------------------------------------
        # Medical Use_Case compatibility
        # ----------------------------------------------------

        matched_conditions = []

        for condition in required_conditions:

            if use_case_matches(
                condition,
                row["Use_Case"]
            ):

                matched_conditions.append(
                    normalize_text(condition)
                )

        # ----------------------------------------------------
        # Keep medicine only if it treats
        # at least one required condition
        # ----------------------------------------------------

        if matched_conditions:

            candidates.append({
                "Drug_Name":
                    row["Drug_Name"],

                "Generic_Name":
                    row["Generic_Name"],

                "Use_Case":
                    row["Use_Case"],

                "Typical_Dosage":
                    row["Typical_Dosage"],

                "Quantity_in_Kit":
                    row["Quantity_in_Kit"],

                "Region_Notes":
                    row["Region_Notes"],

                "Mission_Type":
                    row["Mission_Type"],

                "matched_conditions":
                    matched_conditions
            })

    return pd.DataFrame(
        candidates
    )


# ============================================================
# 7. OPTIMIZE DRUG KIT
# ============================================================

def optimize_drug_kit(
    df,
    region,
    mission,
    required_conditions
):
    """
    Optimize medicine selection.

    Objective:
        1. Cover as many required conditions as possible.
        2. Use the minimum number of medicines.

    NOTE:
        Actual weight optimization will be added later
        when reliable weight data is available.
    """

    # --------------------------------------------------------
    # Normalize conditions
    # --------------------------------------------------------

    normalized_conditions = []

    for condition in required_conditions:

        condition = normalize_text(
            condition
        )

        if condition:
            normalized_conditions.append(
                condition
            )

    # Remove duplicates
    normalized_conditions = list(
        dict.fromkeys(
            normalized_conditions
        )
    )

    if not normalized_conditions:

        return {
            "success": False,
            "region": region,
            "mission": mission,
            "selected_medicines": [],
            "selected_details": [],
            "coverage": 0.0,
            "covered_conditions": [],
            "uncovered_conditions": [],
            "candidate_count": 0,
            "optimization_status":
                "No medical requirements provided"
        }

    # --------------------------------------------------------
    # Find valid candidates
    # --------------------------------------------------------

    candidates = find_candidate_medicines(
        df=df,
        region=region,
        mission=mission,
        required_conditions=
            normalized_conditions
    )

    if candidates.empty:

        return {
            "success": False,
            "region": region,
            "mission": mission,
            "selected_medicines": [],
            "selected_details": [],
            "coverage": 0.0,
            "covered_conditions": [],
            "uncovered_conditions":
                normalized_conditions,
            "candidate_count": 0,
            "optimization_status":
                "No compatible medicines found"
        }

    # --------------------------------------------------------
    # Greedy Set-Cover Optimization
    # --------------------------------------------------------

    remaining_conditions = set(
        normalized_conditions
    )

    covered_conditions = set()

    selected_medicines = []

    selected_details = []

    while remaining_conditions:

        best_drug = None
        best_matches = set()
        best_row = None

        # ----------------------------------------------------
        # Examine every unique drug
        # ----------------------------------------------------

        for drug_name, group in candidates.groupby(
            "Drug_Name"
        ):

            drug_matches = set()
            first_row = None

            for _, row in group.iterrows():

                if first_row is None:
                    first_row = row

                for condition in row[
                    "matched_conditions"
                ]:

                    if condition in remaining_conditions:

                        drug_matches.add(
                            condition
                        )

            # ------------------------------------------------
            # Choose drug covering the largest
            # number of currently uncovered conditions
            # ------------------------------------------------

            if len(drug_matches) > len(
                best_matches
            ):

                best_drug = drug_name
                best_matches = drug_matches
                best_row = first_row

        # ----------------------------------------------------
        # Nothing else can be covered
        # ----------------------------------------------------

        if (
            best_drug is None
            or not best_matches
        ):
            break

        # ----------------------------------------------------
        # Add drug
        # ----------------------------------------------------

        selected_medicines.append(
            best_drug
        )

        covered_conditions.update(
            best_matches
        )

        remaining_conditions -= (
            best_matches
        )

        # ----------------------------------------------------
        # Save details
        # ----------------------------------------------------

        if best_row is not None:

            selected_details.append({

                "Drug_Name":
                    best_row["Drug_Name"],

                "Generic_Name":
                    best_row["Generic_Name"],

                "Use_Case":
                    best_row["Use_Case"],

                "Typical_Dosage":
                    best_row["Typical_Dosage"],

                "Quantity_in_Kit":
                    best_row["Quantity_in_Kit"],

                "Matched_Conditions":
                    sorted(best_matches)
            })

    # ========================================================
    # 8. CALCULATE COVERAGE
    # ========================================================

    total_conditions = len(
        normalized_conditions
    )

    coverage = (
        len(covered_conditions)
        / total_conditions
        if total_conditions > 0
        else 0.0
    )

    # ========================================================
    # 9. OPTIMIZATION STATUS
    # ========================================================

    if coverage == 1.0:

        optimization_status = (
            "Complete coverage"
        )

    elif coverage > 0:

        optimization_status = (
            "Partial coverage"
        )

    else:

        optimization_status = (
            "No coverage"
        )

    # ========================================================
    # 10. RETURN RESULT
    # ========================================================

    return {

        "success":
            coverage > 0,

        "region":
            region,

        "mission":
            mission,

        "selected_medicines":
            selected_medicines,

        "selected_details":
            selected_details,

        "coverage":
            coverage,

        "covered_conditions":
            sorted(
                covered_conditions
            ),

        "uncovered_conditions":
            sorted(
                remaining_conditions
            ),

        "candidate_count":
            len(candidates),

        "optimization_status":
            optimization_status
    }


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

def print_result(result):

    print()
    print("=" * 65)
    print(
        "MILITARY DRUG KIT OPTIMIZER"
    )
    print("=" * 65)

    print(
        f"Region  : {result['region']}"
    )

    print(
        f"Mission : {result['mission']}"
    )

    print()

    # --------------------------------------------------------
    # Selected medicines
    # --------------------------------------------------------

    print(
        "Selected medicines:"
    )

    if result[
        "selected_medicines"
    ]:

        for drug in result[
            "selected_medicines"
        ]:

            print(
                f" - {drug}"
            )

    else:

        print(
            " - None"
        )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    print()

    print(
        "Coverage:",
        f"{result['coverage'] * 100:.2f}%"
    )

    print(
        "Covered:",
        result[
            "covered_conditions"
        ]
    )

    print(
        "Uncovered:",
        result[
            "uncovered_conditions"
        ]
    )

    # --------------------------------------------------------
    # Candidate count
    # --------------------------------------------------------

    print(
        "Candidate medicines:",
        result[
            "candidate_count"
        ]
    )

    print(
        "Medicines selected:",
        len(
            result[
                "selected_medicines"
            ]
        )
    )

    print(
        "Optimization status:",
        result[
            "optimization_status"
        ]
    )

    # --------------------------------------------------------
    # Detailed medicines
    # --------------------------------------------------------

    print()
    print(
        "Selected medicine details:"
    )

    for item in result.get(
        "selected_details",
        []
    ):

        print(
            f"\n - {item['Drug_Name']}"
        )

        print(
            f"   Generic: "
            f"{item['Generic_Name']}"
        )

        print(
            f"   Use Case: "
            f"{item['Use_Case']}"
        )

        print(
            f"   Dosage: "
            f"{item['Typical_Dosage']}"
        )

        print(
            f"   Quantity: "
            f"{item['Quantity_in_Kit']}"
        )

        print(
            f"   Covers: "
            f"{item['Matched_Conditions']}"
        )

    print()
    print("=" * 65)


# ============================================================
# 12. TEST
# ============================================================

if __name__ == "__main__":

    df = load_dataset()

    # --------------------------------------------------------
    # REAL Use_Case values from your dataset
    # --------------------------------------------------------
    #
    # Pain, fever -> Paracetamol
    # Dehydration -> Oral Rehydration Salts
    # Heat cramps risk -> Sodium Chloride Tabs
    #
    # These are actual Use_Case entries in your dataset.
    # --------------------------------------------------------

    result = optimize_drug_kit(

        df=df,

        region="Arid heat",

        mission="Desert Ops",

        required_conditions=[
            "Pain, fever",
            "Dehydration",
            "Heat cramps risk"
        ]
    )

    print_result(
        result
    )