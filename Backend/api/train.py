import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils.constant import DATASET, RESULT_DIR


class RFTrainer:
    def __init__(self, data):
        self.data = data

    def train_dataset(self):
        df = self.data.copy()
        X = df.drop("Gallstone Status", axis=1)
        y = df["Gallstone Status"]

        numeric_cols = [
            'Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
            'Extracellular Water (ECW)', 'Intracellular Water (ICW)', 'Extracellular Fluid/Total Body Water (ECF/TBW)',
            'Total Body Fat Ratio (TBFR) (%)', 'Lean Mass (LM) (%)', 'Body Protein Content (Protein) (%)',
            'Visceral Fat Rating (VFR)', 'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)',
            'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Visceral Muscle Area (VMA) (Kg)',
            'Hepatic Fat Accumulation (HFA)', 'Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
            'High Density Lipoprotein (HDL)', 'Triglyceride', 'Aspartat Aminotransferaz (AST)',
            'Alanin Aminotransferaz (ALT)', 'Alkaline Phosphatase (ALP)', 'Creatinine',
            'Glomerular Filtration Rate (GFR)', 'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D'
        ]

        categorical_cols = [
            "Age", "Gender", "Comorbidity", "Coronary Artery Disease (CAD)",
            "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)"
        ]
    
        feature_cols = categorical_cols + numeric_cols

        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X, y, scaler, feature_cols

    def train_rf(self, X, y):
        param_grid = {
            "n_estimators": [100, 200, 500],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scoring = {
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc"
        }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        return (
            grid_search.best_estimator_,
            grid_search.best_params_,
            grid_search.best_score_,
            grid_search.cv_results_
        )


def get_next_version(result_dir: str, base_name: str = "rf"):
    """Find next available version number for model saving."""
    version = 1
    while True:
        model_path = os.path.join(result_dir, f"{base_name}_v{version}.pkl")
        if not os.path.exists(model_path):
            return version
        version += 1


def train_rf_pipeline():
    os.makedirs(RESULT_DIR, exist_ok=True)

    data = pd.read_csv(DATASET)
    trainer = RFTrainer(data)

    print("Preparing dataset...")
    X, y, scaler, feature_cols = trainer.train_dataset()

    print("Running GridSearchCV...")
    best_model, best_params, best_score, cv_results = trainer.train_rf(X, y)

    # Versioning
    version = get_next_version(RESULT_DIR)

    # Save model
    model_path = os.path.join(RESULT_DIR, f"rf_v{version}.pkl")
    joblib.dump(best_model, model_path)

    # Save scaler
    scaler_path = os.path.join(RESULT_DIR, f"scaler_v{version}.pkl")
    joblib.dump(scaler, scaler_path)

    # Save feature columns
    feature_cols_path = os.path.join(RESULT_DIR, f"feature_cols_v{version}.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f)

    result = {
        "status": "success",
        "version": version,
        "best_params": best_params,
        "best_score": best_score,
        "saved_files": {
            "model": model_path,
            "scaler": scaler_path,
            "feature_cols": feature_cols_path,
        }
    }

    return result


if __name__ == "__main__":
    print(json.dumps(train_rf_pipeline(), indent=4))
