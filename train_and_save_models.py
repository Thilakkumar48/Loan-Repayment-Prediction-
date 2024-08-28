import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

def load_data():
    df = pd.read_csv("loan_data.csv")
    df['purpose'] = LabelEncoder().fit_transform(df['purpose'])
    
    X = df.drop('not.fully.paid', axis=1)
    y = df['not.fully.paid']
    
    return train_test_split(X, y, test_size=0.3, random_state=101)

def train_and_save_models():
    X_train, X_test, y_train, y_test = load_data()
    
    # Train and save Decision Tree model
    dt_clf = DecisionTreeClassifier(max_depth=2)
    dt_clf.fit(X_train, y_train)
    joblib.dump(dt_clf, 'models/decision_tree.pkl')
    
    # Train and save Random Forest model
    rf_clf = RandomForestClassifier(n_estimators=600)
    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, 'models/random_forest.pkl')
    
    # Train and save Gradient Boosting model
    gb_clf = GradientBoostingClassifier(learning_rate=0.05)
    gb_clf.fit(X_train, y_train)
    joblib.dump(gb_clf, 'models/gradient_boosting.pkl')

if __name__ == "__main__":
    train_and_save_models()
