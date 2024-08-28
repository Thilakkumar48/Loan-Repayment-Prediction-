from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load and preprocess data
def load_data():
    df = pd.read_csv("loan_data.csv")
    df['purpose'] = LabelEncoder().fit_transform(df['purpose'])
    
    # Feature and target variables
    X = df.drop('not.fully.paid', axis=1)
    y = df['not.fully.paid']
    
    # Ensure to match feature names
    feature_names = X.columns
    
    return X, y, feature_names, train_test_split(X, y, test_size=0.3, random_state=101)

# Model training
def train_models(X_train, y_train):
    models = {}

    # Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=2)
    dt_clf.fit(X_train, y_train)
    models['decision_tree'] = dt_clf
    
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=600)
    rf_clf.fit(X_train, y_train)
    models['random_forest'] = rf_clf
    
    # Gradient Boosting
    gb_clf = GradientBoostingClassifier(learning_rate=0.05)
    gb_clf.fit(X_train, y_train)
    models['gradient_boosting'] = gb_clf
    
    return models

# Initialize data and models
X, y, feature_names, (X_train, X_test, y_train, y_test) = load_data()
models = train_models(X_train, y_train)

# Save models
joblib.dump(models['decision_tree'], 'models/decision_tree.pkl')
joblib.dump(models['random_forest'], 'models/random_forest.pkl')
joblib.dump(models['gradient_boosting'], 'models/gradient_boosting.pkl')

# Web application routes
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, model_name: str = Form(...), 
            fico: float = Form(...), int_rate: float = Form(...), 
            installment: float = Form(...), log_annual_inc: float = Form(...),
            dti: float = Form(...), purpose: int = Form(...), 
            revol_util: float = Form(...), inq_last_6mths: int = Form(...)):

    if model_name not in models:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model not found"})

    features = np.array([fico, int_rate, installment, log_annual_inc, dti, purpose, revol_util, inq_last_6mths]).reshape(1, -1)

    if features.shape[1] != len(feature_names):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Feature mismatch"})

    model = models[model_name]
    prediction = model.predict(features)
    
    return templates.TemplateResponse("result.html", {"request": request, "prediction": int(prediction[0])})

@app.get("/evaluate/{model_name}", response_class=HTMLResponse)
def evaluate_model(request: Request, model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[model_name]
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    
    return templates.TemplateResponse("result.html", {
        "request": request, 
        "model_name": model_name,
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    })

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
