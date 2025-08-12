import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json


# =========================
#  TRAIN BASE RANDOM FOREST
# =========================
def train_base_rf(X, y):
    """Trains a basic RandomForestClassifier."""
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf


# =========================
#  TRAIN RF WITH GRID SEARCH
# =========================
def train_rf_gridsearch_smart(X_train, y_train):
    """
    Trains a RandomForestClassifier using GridSearchCV to find the best hyperparameters.
    """
    # Parameter grid (balanced between thorough search and speed)
    param_grid = {
        "n_estimators": [100, 200, 500],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"]
    }

    # Stratified CV for robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define multiple scoring metrics, but optimize for f1-score
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

    grid_search.fit(X_train, y_train)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
        grid_search.cv_results_
    )


# =========================
#  EVALUATION
# =========================
def evaluate_rf(X, y_true, model, plot_dir=None, model_name="RF_Base"):
    """Evaluates the model and generates plots for ROC curve and confusion matrix."""
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, labels),
        "precision": precision_score(y_true, labels, pos_label=0),
        "recall": recall_score(y_true, labels, pos_label=0),
        "f1_score": f1_score(y_true, labels, pos_label=0),
        "roc_auc_score": roc_auc_score(y_true, probs),
    }

    cm = confusion_matrix(y_true, labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(y_true, labels, output_dict=True, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, probs)
    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

        # ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f'ROC (AUC = {metrics["roc_auc_score"]:.2f})')
        plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, f"{model_name}_ROC.png"))
        plt.close()

        # Confusion Matrix
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(plot_dir, f"{model_name}_ConfusionMatrix.png"))
        plt.close()

    return metrics


# =========================
#  SAVE METRICS
# =========================
def save_metrics(metrics, file_path, model_name):
    """Saves evaluation metrics to a CSV file."""
    df = pd.DataFrame([metrics])
    df.insert(0, "model_name", model_name)
    df.insert(1, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(file_path, index=False)


# =========================
#  FEATURE IMPORTANCE PLOT
# =========================
def plot_feature_importance(model, X_train, plot_dir, model_name):
    """Generates and saves a feature importance plot."""
    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{model_name}_FeatureImportance.png"))
    plt.close()

    return importance_df


# =======================
#  DATASET FUNCTIONS
# =======================
def original_dataset(data):
    """
    Prepares the original dataset, splits it, scales numeric features,
    and returns the scaler and feature lists for later use.
    """
    df = data.copy()
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
    
    # Combine feature lists to get the final column order
    feature_cols = categorical_cols + numeric_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_cols], y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and fit the scaler ONLY on the training data
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    # Transform the test data using the same scaler
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Return the scaler and column lists along with the datasets
    return X_train, X_test, y_train, y_test, scaler, feature_cols


# =========================
#  MAIN
# =========================
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('https://raw.githubusercontent.com/Bhawesh-Agrawal/Gallstone_Risk_Stratification/master/Dataset/gallstone_.csv')

    # Define directory for results
    results_dir = "results_rf"
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, "metrics.csv")

    # --- TRAIN AND SAVE BEST MODEL ---
    print("Preparing dataset...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = original_dataset(data)

    print("Starting GridSearchCV to find the best model...")
    best_model, best_params, best_score, cv_results = train_rf_gridsearch_smart(X_train, y_train)
    
    print("\n--- Best Model Found ---")
    print(f"Best F1-Score (CV): {best_score:.4f}")
    print("Best Parameters:")
    print(best_params)

    # Evaluate the final model on the held-out test set
    print("\nEvaluating the best model on the test set...")
    metrics = evaluate_rf(X_test, y_test, best_model, plot_dir=results_dir, model_name="RF_Best_Final")
    metrics["best_params"] = best_params
    metrics["best_cv_score"] = best_score
    save_metrics(metrics, metrics_file, "RF_Best_Final")
    print("Evaluation metrics saved.")

    # Plot and save feature importance for the best model
    feature_importance_df = plot_feature_importance(best_model, X_train, results_dir, "RF_Best_Final")
    print("Feature importance plot saved.")

    # --- SAVE ARTIFACTS FOR PREDICTION ---
    # Save the trained model
    model_path = os.path.join(results_dir, "rf_best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved at: {model_path}")

    # Save the scaler
    scaler_path = os.path.join(results_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved at: {scaler_path}")

    # Save the feature column list
    feature_cols_path = os.path.join(results_dir, "feature_cols.json")
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_cols, f)
    print(f"✅ Feature columns saved at: {feature_cols_path}")

