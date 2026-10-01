#train_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess

def train_and_save_model():
    X_train, X_test, y_train, y_test, encoders, label_encoder = load_and_preprocess()

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save model + encoders
    with open("../models/model.pkl", "wb") as f:
        pickle.dump((model, encoders, label_encoder), f)

    print("✅ Model trained and saved at ../models/model.pkl")

if __name__ == "__main__":
    train_and_save_model()
