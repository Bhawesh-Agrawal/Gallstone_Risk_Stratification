import joblib
import pandas as pd
import numpy as np
import shap
import json
import os
import warnings

# Suppress a common warning from SHAP with certain scikit-learn versions
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")

class GallstonePredictor:
    """
    A class to load a pre-trained RandomForest model and its associated scaler
    to predict gallstone risk and provide SHAP-based explanations.
    """
    def __init__(self, model_path, scaler_path, feature_cols_path):
        """
        Initializes the predictor by loading the model, scaler, and feature list.

        Args:
            model_path (str): Path to the saved .pkl model file.
            scaler_path (str): Path to the saved .pkl scaler file.
            feature_cols_path (str): Path to the saved .json file with the list of feature names.
        """
        if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_cols_path]):
            raise FileNotFoundError("One or more required files (model, scaler, or feature_cols) not found.")
            
        print("Loading model and associated artifacts...")
        # Load the pre-trained model, scaler, and feature columns
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_cols_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        # Identify numeric columns by checking which columns the scaler was fitted on
        self.numeric_cols = [col for col in self.feature_cols if col in self.scaler.feature_names_in_]
        
        # Initialize a SHAP explainer for the model
        # TreeExplainer is efficient for tree-based models like RandomForest
        self.explainer = shap.TreeExplainer(self.model)
        print("Predictor initialized successfully.")

    def predict_and_explain(self, input_data):
        """
        Makes a prediction on a single instance of input data and performs
        a SHAP analysis to explain the prediction.

        The SHAP analysis identifies the top 5 features contributing towards a
        positive prediction and the top 5 features contributing towards a negative one.

        Args:
            input_data (dict): A dictionary where keys are feature names and
                               values are the corresponding feature values.

        Returns:
            str: A JSON string containing the prediction, probabilities, and
                 a detailed SHAP analysis with top contributors.
        """
        # Convert the input dictionary to a pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns from training are present in the DataFrame,
        # filling missing ones with 0 and reordering to match the model's expectation.
        input_df = input_df.reindex(columns=self.feature_cols, fill_value=0)

        # Create a copy for scaling to avoid SettingWithCopyWarning
        df_scaled = input_df.copy()
        
        # Scale the numeric features using the loaded scaler
        df_scaled[self.numeric_cols] = self.scaler.transform(df_scaled[self.numeric_cols])

        # --- Prediction ---
        # Get prediction probabilities for both classes [P(class 0), P(class 1)]
        prediction_proba = self.model.predict_proba(df_scaled)[0]
        # Determine the predicted class (0=Positive, 1=Negative) based on the higher probability
        prediction = np.argmax(prediction_proba)

        # --- SHAP Analysis using modern Explanation object for robustness ---
        explanation = self.explainer(df_scaled)
        shap_values_all_classes = explanation.values[0]
        base_values_all_classes = explanation.base_values[0]

        # --- Get SHAP values specifically for the positive class (class 0) ---
        if shap_values_all_classes.ndim == 2 and shap_values_all_classes.shape[1] == 2:
            # Case: values for both classes are returned.
            shap_values_for_class_0 = shap_values_all_classes[:, 0]
            base_value_for_class_0 = base_values_all_classes[0]
        else:
            # Case: values for only the positive class (class 1) are returned.
            # SHAP values for class 0 are the negative of those for class 1.
            shap_values_for_class_0 = -shap_values_all_classes
            base_value_for_class_0 = -base_values_all_classes

        # Create a DataFrame of features and their SHAP contributions to class 0
        contributions_df = pd.DataFrame({
            'feature': self.feature_cols,
            'shap_value_class_0': shap_values_for_class_0
        })

        # --- Sort to find top contributors ---
        # Top 5 contributors TOWARDS a positive prediction (highest positive SHAP values for class 0)
        top_positive_contributors = contributions_df.sort_values(
            by='shap_value_class_0', ascending=False
        ).head(5)

        # Top 5 contributors AWAY from a positive prediction (i.e., towards negative)
        # (most negative SHAP values for class 0)
        top_negative_contributors = contributions_df.sort_values(
            by='shap_value_class_0', ascending=True
        ).head(5)

        # --- Format Output ---
        output = {
            "prediction_label": "Positive" if prediction == 0 else "Negative",
            "prediction_value": int(prediction),
            "probability": {
                "positive (class 0)": f"{prediction_proba[0]:.4f}",
                "negative (class 1)": f"{prediction_proba[1]:.4f}"
            },
            "shap_analysis": {
                "base_value_for_positive_class": float(base_value_for_class_0),
                "top_contributors_to_positive": {
                    row.feature: f"{row.shap_value_class_0:.4f}" for index, row in top_positive_contributors.iterrows()
                },
                "top_contributors_to_negative": {
                    row.feature: f"{row.shap_value_class_0:.4f}" for index, row in top_negative_contributors.iterrows()
                }
            }
        }
        
        # Return the result as a nicely formatted JSON string
        return json.dumps(output, indent=4)


# =========================
#  EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    # Define paths to the saved artifacts
    RESULTS_DIR = "results_rf"
    MODEL_PATH = os.path.join(RESULTS_DIR, "rf_best_model.pkl")
    SCALER_PATH = os.path.join(RESULTS_DIR, "scaler.pkl")
    FEATURES_PATH = os.path.join(RESULTS_DIR, "feature_cols.json")

    try:
        # --- Load real data from the original dataset ---
        print("Loading data from the original dataset to create a test case...")
        full_dataset = pd.read_csv('https://raw.githubusercontent.com/Bhawesh-Agrawal/Gallstone_Risk_Stratification/master/Dataset/gallstone_.csv')
        
        # Select a sample row to test (e.g., the 10th patient)
        sample_index = 10
        sample_patient_record = full_dataset.iloc[sample_index]
        
        # Separate the actual outcome from the features
        actual_status = sample_patient_record['Gallstone Status']
        real_patient_data = sample_patient_record.drop('Gallstone Status').to_dict()

        print(f"\n--- Testing with Patient Data from Index {sample_index} ---")
        print(f"Actual Status from Dataset: {'Positive (0)' if actual_status == 0 else 'Negative (1)'}")

        # Initialize the predictor
        predictor = GallstonePredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            feature_cols_path=FEATURES_PATH
        )
        
        # Get the prediction and explanation for the sample data
        print("\n--- Performing Prediction and SHAP Analysis ---")
        prediction_json = predictor.predict_and_explain(real_patient_data)

        # Print the final JSON output
        print("\n--- Results ---")
        print(prediction_json)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure you have run the 'train_and_save_artifacts.py' script first to generate the necessary files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

