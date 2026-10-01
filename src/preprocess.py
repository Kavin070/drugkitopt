#preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(file_path="../data/dataset.csv"):
    df = pd.read_csv(file_path)

    # Drop duplicates, handle missing values
    df = df.drop_duplicates().dropna()

    # Features & Target
    X = df[["Use_Case", "Region_Notes", "Mission_Type"]]
    y = df["Drug_Name"]

    # Encode categorical variables
    encoders = {}
    for col in X.columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, encoders, label_encoder
