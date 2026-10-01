import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


DATA_PATH = "../data/dataset.csv"


def main():

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    df = df.drop_duplicates().dropna()

    scenario_cols = [
        "Use_Case",
        "Region_Notes",
        "Mission_Type",
    ]

    # ---------------------------------------------------------
    # 2. Create scenario-level dataset
    # ---------------------------------------------------------
    scenarios = (
        df.groupby(scenario_cols)["Drug_Name"]
        .apply(list)
        .reset_index(name="ground_truth")
    )

    print("Total scenarios:", len(scenarios))

    # ---------------------------------------------------------
    # 3. Find Use_Cases with enough scenarios
    # ---------------------------------------------------------
    use_case_counts = (
        scenarios.groupby("Use_Case")
        .size()
    )

    eligible_use_cases = use_case_counts[
        use_case_counts >= 2
    ].index

    scenarios = scenarios[
        scenarios["Use_Case"].isin(
            eligible_use_cases
        )
    ].copy()

    print(
        "Eligible scenarios:",
        len(scenarios)
    )

    # ---------------------------------------------------------
    # 4. HOLD OUT COMPLETE USE CASES
    #
    # This is the important difference.
    #
    # The model will NOT see these Use_Cases during training.
    # ---------------------------------------------------------
    unique_use_cases = (
        scenarios["Use_Case"]
        .drop_duplicates()
        .sample(
            frac=1,
            random_state=42
        )
        .tolist()
    )

    split_index = int(
        len(unique_use_cases) * 0.8
    )

    train_use_cases = set(
        unique_use_cases[:split_index]
    )

    test_use_cases = set(
        unique_use_cases[split_index:]
    )

    train_scenarios = scenarios[
        scenarios["Use_Case"].isin(
            train_use_cases
        )
    ].copy()

    test_scenarios = scenarios[
        scenarios["Use_Case"].isin(
            test_use_cases
        )
    ].copy()

    print(
        "Training Use_Cases:",
        len(train_use_cases)
    )

    print(
        "Testing Use_Cases:",
        len(test_use_cases)
    )

    print(
        "Training scenarios:",
        len(train_scenarios)
    )

    print(
        "Testing scenarios:",
        len(test_scenarios)
    )

    # ---------------------------------------------------------
    # 5. Convert scenarios back to original rows
    # ---------------------------------------------------------
    train_df = df[
        df["Use_Case"].isin(
            train_use_cases
        )
    ].copy()

    test_df = df[
        df["Use_Case"].isin(
            test_use_cases
        )
    ].copy()

    # ---------------------------------------------------------
    # 6. Features
    # ---------------------------------------------------------
    feature_columns = [
        "Use_Case",
        "Region_Notes",
        "Mission_Type",
    ]

    encoders = {}

    for column in feature_columns:

        encoder = LabelEncoder()

        # IMPORTANT:
        # Fit only on training data.
        encoder.fit(
            train_df[column]
        )

        train_df[column] = encoder.transform(
            train_df[column]
        )

        # Test may contain unseen values.
        # Handle unseen categories safely.
        test_values = []

        for value in test_df[column]:

            if value in encoder.classes_:

                test_values.append(
                    encoder.transform(
                        [value]
                    )[0]
                )

            else:

                test_values.append(-1)

        test_df[column] = test_values

        encoders[column] = encoder

    # ---------------------------------------------------------
    # 7. Drug labels
    # ---------------------------------------------------------
    drug_encoder = LabelEncoder()

    drug_encoder.fit(
        train_df["Drug_Name"]
    )

    y_train = drug_encoder.transform(
        train_df["Drug_Name"]
    )

    X_train = train_df[
        feature_columns
    ]

    # ---------------------------------------------------------
    # 8. Train Random Forest
    # ---------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # 9. Scenario-level evaluation
    # ---------------------------------------------------------
    scenario_hits = []

    y_true = []
    y_pred = []

    unknown_drug_count = 0

    for _, scenario in test_scenarios.iterrows():

        use_case = scenario["Use_Case"]
        region = scenario["Region_Notes"]
        mission = scenario["Mission_Type"]

        # Encode Use_Case
        if use_case in encoders["Use_Case"].classes_:
            encoded_use_case = (
                encoders["Use_Case"]
                .transform([use_case])[0]
            )
        else:
            encoded_use_case = -1

        # Encode Region
        if region in encoders["Region_Notes"].classes_:
            encoded_region = (
                encoders["Region_Notes"]
                .transform([region])[0]
            )
        else:
            encoded_region = -1

        # Encode Mission
        if mission in encoders["Mission_Type"].classes_:
            encoded_mission = (
                encoders["Mission_Type"]
                .transform([mission])[0]
            )
        else:
            encoded_mission = -1

        X_input = pd.DataFrame(
            [{
                "Use_Case": encoded_use_case,
                "Region_Notes": encoded_region,
                "Mission_Type": encoded_mission
            }]
        )

        prediction = model.predict(
            X_input
        )[0]

        predicted_drug = (
            drug_encoder
            .inverse_transform(
                [prediction]
            )[0]
        )

        ground_truth = set(
            scenario["ground_truth"]
        )

        hit = (
            predicted_drug
            in ground_truth
        )

        scenario_hits.append(hit)

        # Classification metrics
        for actual_drug in ground_truth:

            if actual_drug in drug_encoder.classes_:

                y_true.append(
                    actual_drug
                )

                y_pred.append(
                    predicted_drug
                )

            else:

                unknown_drug_count += 1

    # ---------------------------------------------------------
    # 10. Results
    # ---------------------------------------------------------
    total = len(
        scenario_hits
    )

    correct = sum(
        scenario_hits
    )

    incorrect = (
        total - correct
    )

    hit_rate = (
        correct / total
        if total > 0
        else 0
    )

    print()
    print("=" * 65)
    print("ROBUST SCENARIO EVALUATION")
    print("=" * 65)

    print(
        f"Training Use_Cases : {len(train_use_cases)}"
    )

    print(
        f"Testing Use_Cases  : {len(test_use_cases)}"
    )

    print(
        f"Training scenarios : {len(train_scenarios)}"
    )

    print(
        f"Testing scenarios  : {len(test_scenarios)}"
    )

    print()

    print(
        f"Scenario Hit Rate  : {hit_rate:.4f}"
    )

    print(
        f"Scenario Hit Rate  : {hit_rate * 100:.2f}%"
    )

    print(
        f"Correct scenarios  : {correct}"
    )

    print(
        f"Incorrect scenarios: {incorrect}"
    )

    print()

    if len(y_true) > 0:

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        print(
            f"Drug Accuracy       : {accuracy:.4f}"
        )

        print(
            f"Drug Precision      : {precision:.4f}"
        )

        print(
            f"Drug Recall         : {recall:.4f}"
        )

        print(
            f"Drug F1 Score       : {f1:.4f}"
        )

    print()

    print(
        "Unknown test drugs  :",
        unknown_drug_count
    )

    print("=" * 65)


if __name__ == "__main__":
    main()