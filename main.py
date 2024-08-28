from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

app = FastAPI()

# Load models and encoders
try:
    decision_tree_model = joblib.load('decision_tree_model.pkl')
    random_forest_model = joblib.load('random_forest_model.pkl')
    gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Model file not found: {e}")

# Load and preprocess data
df = pd.read_csv("loan_data.csv")
label_encoder = LabelEncoder()
df['purpose'] = label_encoder.fit_transform(df['purpose'])

# Define data structure for prediction requests
class LoanRequest(BaseModel):
    fico: float
    int_rate: float
    credit_policy: int
    purpose: str

# Endpoint for prediction
@app.post("/predict/")
def predict(loan: LoanRequest):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([{
        'fico': loan.fico,
        'int.rate': loan.int_rate,
        'credit.policy': loan.credit_policy,
        'purpose': label_encoder.transform([loan.purpose])[0]
    }])
    
    # Predict with each model
    try:
        dt_prediction = decision_tree_model.predict(input_data)[0]
        rf_prediction = random_forest_model.predict(input_data)[0]
        gb_prediction = gradient_boosting_model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "Decision Tree Prediction": int(dt_prediction),
        "Random Forest Prediction": int(rf_prediction),
        "Gradient Boosting Prediction": int(gb_prediction)
    }

# Endpoint for model evaluation
@app.get("/evaluate/")
def evaluate():
    # Load and preprocess data
    df = pd.read_csv("loan_data.csv")
    df['purpose'] = label_encoder.transform(df['purpose'])  # Ensure consistent encoding
    X = df.drop('not.fully.paid', axis=1)
    y = df['not.fully.paid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Evaluate models
    dt_predictions = decision_tree_model.predict(X_test)
    rf_predictions = random_forest_model.predict(X_test)
    gb_predictions = gradient_boosting_model.predict(X_test)

    dt_accuracy = accuracy_score(y_test, dt_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    gb_accuracy = accuracy_score(y_test, gb_predictions)

    return {
        "Decision Tree Accuracy": dt_accuracy,
        "Random Forest Accuracy": rf_accuracy,
        "Gradient Boosting Accuracy": gb_accuracy
    }
