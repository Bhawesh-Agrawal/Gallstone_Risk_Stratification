import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    make_scorer
)
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode


# =========================
#   MODEL TRAINING
# =========================
def train_kmeans(X, n_clusters, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X)
    return km


# =========================
#   LABEL ALIGNMENT
# =========================
def align_cluster_labels(y_true, cluster_labels):
    """
    Aligns KMeans cluster labels to match the majority vote with y_true.
    This avoids label flipping issues.
    """
    new_labels = np.zeros_like(cluster_labels)
    for cluster in np.unique(cluster_labels):
        mask = (cluster_labels == cluster)
        new_labels[mask] = mode(y_true[mask])[0]
    return new_labels


# =========================
#   EVALUATION
# =========================
def evaluate_kmeans(y_true, km, plot_dir=None, model_name="KMeans"):
    labels = align_cluster_labels(y_true.values, km.labels_)

    metrics = {
        'accuracy': accuracy_score(y_true, labels),
        'precision': precision_score(y_true, labels, pos_label=0),
        'recall': recall_score(y_true, labels, pos_label=0),
        'f1_score': f1_score(y_true, labels, pos_label=0),
    }

    # ROC AUC requires probability or continuous scores; 
    # Since KMeans doesn't provide this, we treat labels as scores (imperfect).
    try:
        metrics['roc_auc_score'] = roc_auc_score(y_true, labels)
        fpr, tpr, _ = roc_curve(y_true, labels)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    except ValueError:
        # If ROC AUC calculation fails (e.g., only one class present), skip
        metrics['roc_auc_score'] = None
        metrics['roc_curve'] = {'fpr': [], 'tpr': []}

    # Confusion Matrix
    cm = confusion_matrix(y_true, labels)
    metrics['confusion_matrix'] = cm.tolist()

    # Classification Report
    metrics['classification_report'] = classification_report(y_true, labels, output_dict=True, zero_division=0)

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

        # ROC Curve Plot (only if valid)
        if metrics['roc_auc_score'] is not None:
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {metrics["roc_auc_score"]:.2f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plot_dir, f'{model_name}_ROC.png'))
            plt.close()

        # Confusion Matrix Plot
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
#   ELBOW METHOD PLOT
# =========================
def elbow_plot(X, n=10, plot_dir=None):
    cs = []
    for i in range(1, n + 1):
        kme = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kme.fit(X)
        cs.append(kme.inertia_)

    plt.plot(range(1, n + 1), cs, marker='o')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia (WCSS)')
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'ElbowPlot.png'))
    plt.close()


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
#   Without Feature Engineering
# =========================
def original_dataset(data):
    df = data.copy()
    X = df.drop("Gallstone Status", axis=1)
    y = df["Gallstone Status"]

    cols = [
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

    cols2 = [
        "Age", "Gender", "Comorbidity", "Coronary Artery Disease (CAD)",
        "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)"
    ]

    ms = MinMaxScaler()
    x = ms.fit_transform(X[cols])
    x = pd.DataFrame(x, columns=cols)
    X = pd.concat([X[cols2], x], axis=1)

    return X, y


# =========================
#   With Feature Engineering
# =========================
def featured_dataset(data):
    df = data.copy()

    df['BMI_CRP_Product'] = df['Body Mass Index (BMI)'] * df['C-Reactive Protein (CRP)']
    df['Muscle_Fat_Ratio'] = df['Muscle Mass (MM)'] / df['Total Fat Content (TFC)']

    feature_cols = ['Gallstone Status',
                    'Age', 'Gender', 'Comorbidity', 'Coronary Artery Disease (CAD)',
                    'Hypothyroidism', 'Hyperlipidemia', 'Diabetes Mellitus (DM)',
                    'Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
                    'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Glucose',
                    'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
                    'High Density Lipoprotein (HDL)', 'Triglyceride', 'Aspartat Aminotransferaz (AST)',
                    'C-Reactive Protein (CRP)', 'Vitamin D', 'BMI_CRP_Product', 'Muscle_Fat_Ratio']

    X = df[feature_cols].drop("Gallstone Status", axis=1)
    y = df["Gallstone Status"]

    ns = MinMaxScaler()
    x = ns.fit_transform(X[['Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
                           'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Glucose',
                           'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)',
                           'Triglyceride', 'Aspartat Aminotransferaz (AST)', 'C-Reactive Protein (CRP)',
                           'Vitamin D', 'BMI_CRP_Product', 'Muscle_Fat_Ratio']])

    x = pd.DataFrame(x, columns=['Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
                                'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Glucose',
                                'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)',
                                'Triglyceride', 'Aspartat Aminotransferaz (AST)', 'C-Reactive Protein (CRP)',
                                'Vitamin D', 'BMI_CRP_Product', 'Muscle_Fat_Ratio'])

    cols2 = [
        "Age", "Gender", "Comorbidity", "Coronary Artery Disease (CAD)",
        "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)"
    ]

    X = pd.concat([X[cols2], x], axis=1)

    return X, y


# =========================
#   PIPELINE EXAMPLE
# =========================
if __name__ == "__main__":

    data = pd.read_csv('https://raw.githubusercontent.com/Bhawesh-Agrawal/Gallstone_Risk_Stratification/master/Dataset/gallstone_.csv')

    #X, y = original_dataset(data)
    X, y = featured_dataset(data)

    results_dir = "results_kmeans"
    elbow_plot(X, plot_dir=results_dir)

    km = train_kmeans(X, n_clusters=2)
    metrics = evaluate_kmeans(y, km, plot_dir=results_dir, model_name="KMeans_Featured")

    save_metrics(metrics, os.path.join(results_dir, "metrics.csv"), "KMeans_Featured")
    print("Metrics saved:", metrics)
