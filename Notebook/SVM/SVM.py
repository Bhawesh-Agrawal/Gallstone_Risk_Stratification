import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =========================
#   TRAIN BASE SVM
# =========================
def train_base_svm(X, y, random_state=42):
    svc = SVC(random_state=random_state, probability=True)
    svc.fit(X, y)
    return svc


# =========================
#   TRAIN SVM WITH GRID SEARCH
# =========================
def train_svm_gridsearch(X, y, cv=5, random_state=42):
    param_grid = {
        'C': [0.1, 1, 10, 100],         # Soft margin to hard margin
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']      # Kernel coefficient
    }
    svc = SVC(random_state=random_state, probability=True)
    grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# =========================
#   EVALUATION
# =========================
def evaluate_svm(X, y_true, model, plot_dir=None, model_name="SVM_Base"):
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, labels),
        'precision': precision_score(y_true, labels, pos_label=0),
        'recall': recall_score(y_true, labels, pos_label=0),
        'f1_score': f1_score(y_true, labels, pos_label=0),
        'roc_auc_score': roc_auc_score(y_true, probs),
    }

    cm = confusion_matrix(y_true, labels)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['classification_report'] = classification_report(y_true, labels, output_dict=True, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, probs)
    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {metrics["roc_auc_score"]:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, f'{model_name}_ROC.png'))
        plt.close()

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        ticks = np.arange(len(np.unique(y_true)))
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(plot_dir, f'{model_name}_ConfusionMatrix.png'))
        plt.close()

    return metrics


# =========================
#   SAVE METRICS
# =========================
def save_metrics(metrics, file_path, model_name):
    df = pd.DataFrame([metrics])
    df.insert(0, "model_name", model_name)
    df.insert(1, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(file_path, index=False)


# =========================
#   ORIGINAL DATASET
# =========================
def original_dataset(data):
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

    X_train, X_test, y_train, y_test = train_test_split(
        X[categorical_cols + numeric_cols], y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test


# =======================
#   FEATURED DATASET
# =======================
def featured_dataset(data):
    df = data.copy()

    df['BMI_CRP_Product'] = df['Body Mass Index (BMI)'] * df['C-Reactive Protein (CRP)']
    df['Muscle_Fat_Ratio'] = df['Muscle Mass (MM)'] / df['Total Fat Content (TFC)']

    feature_cols = [
        'Age', 'Gender', 'Comorbidity', 'Coronary Artery Disease (CAD)',
        'Hypothyroidism', 'Hyperlipidemia', 'Diabetes Mellitus (DM)',
        'Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
        'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Glucose',
        'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
        'High Density Lipoprotein (HDL)', 'Triglyceride', 'Aspartat Aminotransferaz (AST)',
        'C-Reactive Protein (CRP)', 'Vitamin D', 'BMI_CRP_Product', 'Muscle_Fat_Ratio'
    ]

    X = df[feature_cols]
    y = df["Gallstone Status"]

    numeric_cols = [
        'Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
        'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Glucose',
        'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
        'High Density Lipoprotein (HDL)', 'Triglyceride', 'Aspartat Aminotransferaz (AST)',
        'C-Reactive Protein (CRP)', 'Vitamin D', 'BMI_CRP_Product', 'Muscle_Fat_Ratio'
    ]
    categorical_cols = [
        'Age', 'Gender', 'Comorbidity', 'Coronary Artery Disease (CAD)',
        'Hypothyroidism', 'Hyperlipidemia', 'Diabetes Mellitus (DM)'
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test


# =========================
#   MAIN
# =========================
if __name__ == "__main__":
    data = pd.read_csv('https://raw.githubusercontent.com/Bhawesh-Agrawal/Gallstone_Risk_Stratification/master/Dataset/gallstone_.csv')

    results_dir = "results_svm"
    metrics_file = os.path.join(results_dir, "metrics.csv")

    # ORIGINAL DATASET TRAINING
    #X_train, X_test, y_train, y_test = original_dataset(data)
    #model = train_base_svm(X_train, y_train)
    #metrics = evaluate_svm(X_test, y_test, model, plot_dir=results_dir, model_name="SVM_Original")
    #save_metrics(metrics, metrics_file, "SVM_Original")

    # FEATURED DATASET TRAINING
    X_train, X_test, y_train, y_test = featured_dataset(data)
    #model = train_base_svm(X_train, y_train)
    #metrics = evaluate_svm(X_test, y_test, model, plot_dir=results_dir, model_name="SVM_Featured")
    #save_metrics(metrics, metrics_file, "SVM_Featured")

    # ORIGINAL DATASET TRAINING (Grid Search)
    #best_model, best_params, best_score = train_svm_gridsearch(X_train, y_train)
    #metrics = evaluate_svm(X_test, y_test, best_model, plot_dir=results_dir, model_name="SVM_Original_grid")
    #metrics["best_params"] = best_params
    #metrics["best_cv_score"] = best_score
    #save_metrics(metrics, metrics_file, "SVM_Original_grid")

    # FEATURED DATASET TRAINING (Grid Search)
    best_model, best_params, best_score = train_svm_gridsearch(X_train, y_train)
    metrics = evaluate_svm(X_test, y_test, best_model, plot_dir=results_dir, model_name="SVM_Featured_grid")
    metrics["best_params"] = best_params
    metrics["best_cv_score"] = best_score
    save_metrics(metrics, metrics_file, "SVM_Featured_grid")

    print("Base SVM metrics saved:", metrics)
