#evaluate.py
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_preprocess
from train_model import train_and_save_model
import pickle

def evaluate():
    # Train & save
    train_and_save_model()

    # Reload model
    with open("../models/model.pkl", "rb") as f:
        model, encoders, label_encoder = pickle.load(f)

    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess()

    y_pred = model.predict(X_test)

    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
