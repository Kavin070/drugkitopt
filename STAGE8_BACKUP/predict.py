#predict.py
import pickle
import pandas as pd

def load_model():
    with open("../models/model.pkl", "rb") as f:
        model, encoders, label_encoder = pickle.load(f)
    return model, encoders, label_encoder

def predict_drug(use_case, region, mission):
    model, encoders, label_encoder = load_model()

    # Encode input
    input_data = pd.DataFrame([{
        "Use_Case": encoders["Use_Case"].transform([use_case])[0],
        "Region_Notes": encoders["Region_Notes"].transform([region])[0],
        "Mission_Type": encoders["Mission_Type"].transform([mission])[0]
    }])

    prediction = model.predict(input_data)[0]
    drug_name = label_encoder.inverse_transform([prediction])[0]

    return drug_name
