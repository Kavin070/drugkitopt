from ml_adapter import predict_with_existing_model


def test_ml_adapter_has_explicit_result_contract():
    result = predict_with_existing_model(
        {
            "symptom": "fever",
            "condition": "malaria",
            "region": "tropical",
            "mission_type": "jungle",
            "environment": "humid",
        }
    )

    assert result["source"] == "existing_ml_model"
    assert result["status"] in {"SUCCESS", "UNAVAILABLE", "ERROR"}
