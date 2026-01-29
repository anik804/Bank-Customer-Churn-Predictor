import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def train_model():
    # Load the dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv("Churn_Modelling.csv")
    except FileNotFoundError:
        print("Error: Churn_Modelling.csv not found.")
        return

    # Drop unnecessary columns
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    # Separate X (features) and y (target)
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Identify Numerical & Categorical Columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Create and Fit Pipeline with Gradient Boosting
    gb_model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", gb_model)
    ])

    print("Training Gradient Boosting model...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Saving the model
    model_filename = 'gb_churn_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(pipeline, file)

    print(f"\nTrained model saved to {model_filename}")

if __name__ == "__main__":
    train_model()
