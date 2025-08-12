import json
import os
import pandas as pd
from predictor import GallstonePredictor  # Assuming predictor.py is in the same directory
import google.generativeai as genai
from dotenv import load_dotenv # Import the dotenv library
import time
from tqdm import tqdm # For a nice progress bar

class ReportGenerator:
    """
    A class to generate a detailed, user-friendly prompt and then use a real LLM
    (like Google's Gemini) to generate an explanatory report.
    """
    def __init__(self, api_key):
        """
        Initializes the report generator and configures the LLM API.

        Args:
            api_key (str): The API key for the generative model service.
        """
        if not api_key:
            raise ValueError("API key is required to initialize the ReportGenerator.")
        # Configure the Gemini API with the provided key
        genai.configure(api_key=api_key)
        # Use a fast and capable model suitable for application use
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        print("Gemini API configured successfully.")

    def generate_meta_prompt(self, patient_data, prediction_json):
        """
        Generates a detailed, structured prompt FOR THE LLM to generate a report.
        This is the "instruction" prompt.

        Args:
            patient_data (dict): The original patient data.
            prediction_json (str): The JSON output from the GallstonePredictor.

        Returns:
            str: A detailed prompt for the language model.
        """
        prediction_data = json.loads(prediction_json)
        
        # To make the prompt cleaner, we can format the SHAP values nicely
        positive_contributors = "\n".join([f"- {k}: {v}" for k, v in prediction_data['shap_analysis']['top_contributors_to_positive'].items()])
        negative_contributors = "\n".join([f"- {k}: {v}" for k, v in prediction_data['shap_analysis']['top_contributors_to_negative'].items()])


        prompt = (
            "You are a helpful medical assistant. Your task is to generate a clear, "
            "explanatory report for a patient based on their health data and a machine "
            "learning model's prediction. The report should be easy to understand for "
            "someone without a medical background. It must be structured with clear headings.\n\n"
            "--- PATIENT DATA ---\n"
            f"{json.dumps(patient_data, indent=2)}\n\n"
            "--- MODEL PREDICTION & ANALYSIS ---\n"
            f"Prediction: {prediction_data['prediction_label']} (Probability: {prediction_data['probability']})\n"
            f"Base SHAP Value (Average Risk): {prediction_data['shap_analysis']['base_value_for_positive_class']}\n\n"
            "**Factors Increasing Risk (Positive SHAP Contribution):**\n"
            f"{positive_contributors}\n\n"
            "**Factors Decreasing Risk (Negative SHAP Contribution):**\n"
            f"{negative_contributors}\n\n"
            "--- REPORT GENERATION TASK ---\n"
            "Please generate a report that includes the following sections:\n"
            "1.  **Overall Risk Assessment:** A clear statement of the predicted risk "
            "(Positive or Negative for gallstones) and the model's confidence "
            "(probability). Explain what this means in simple terms.\n"
            "2.  **Key Factors Increasing Your Risk:** Based on the SHAP analysis, explain "
            "the top factors that pushed the prediction towards 'Positive'. Explain "
            "each factor's relevance to gallstone risk in simple, easy-to-understand language.\n"
            "3.  **Key Factors Decreasing Your Risk:** Based on the SHAP analysis, explain "
            "the top factors that pushed the prediction towards 'Negative'. Explain why these "
            "factors are protective.\n"
            "4.  **Important Disclaimer and Recommendations:** Provide general advice based on the "
            "findings. Crucially, state that this is not medical advice and strongly "
            "recommend that the user discusses these results with a healthcare professional.\n"
        )
        return prompt

    def get_llm_report(self, prompt):
        """
        Sends a prompt to the configured Gemini model API and returns the report.

        Args:
            prompt (str): The prompt to send to the language model.

        Returns:
            str: The generated report from the language model, or an error message.
        """
        try:
            # Adding a small delay to respect rate limits
            time.sleep(1.5) # Increased delay to be safer with RPM limits
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"ERROR: An error occurred while communicating with the Gemini API: {e}"


if __name__ == "__main__":
    # Load environment variables from a .env file
    load_dotenv()

    # Define paths to the saved artifacts
    RESULTS_DIR = "results_rf"
    MODEL_PATH = os.path.join(RESULTS_DIR, "rf_best_model.pkl")
    SCALER_PATH = os.path.join(RESULTS_DIR, "scaler.pkl")
    FEATURES_PATH = os.path.join(RESULTS_DIR, "feature_cols.json")
    OUTPUT_FILE = "finetuning_dataset.jsonl"

    try:
        # --- Get API Key from environment ---
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please create a .env file and add it there.")

        # --- Initialize services ---
        report_generator = ReportGenerator(api_key=api_key)
        predictor = GallstonePredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            feature_cols_path=FEATURES_PATH
        )

        # --- Load the full dataset ---
        print("Loading the full dataset...")
        full_dataset = pd.read_csv('https://raw.githubusercontent.com/Bhawesh-Agrawal/Gallstone_Risk_Stratification/master/Dataset/gallstone_.csv')
        
        # --- Check for existing progress and resume ---
        start_index = 0
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                # Count how many lines (records) are already saved
                start_index = sum(1 for line in f)
            print(f"Resuming from record {start_index}. Found {start_index} existing records in {OUTPUT_FILE}.")
        
        print(f"Total records to process: {len(full_dataset)}. Starting from index: {start_index}.")


        # --- Start the generation pipeline ---
        # Open the file in append mode ('a') to add to it instead of overwriting
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            # Iterate from the last saved point
            for index, row in tqdm(full_dataset.iloc[start_index:].iterrows(), total=len(full_dataset) - start_index, desc="Generating Reports"):
                try:
                    # 1. Prepare data for the current row
                    patient_data = row.drop('Gallstone Status').to_dict()

                    # 2. Get the prediction and SHAP analysis
                    prediction_json = predictor.predict_and_explain(patient_data)

                    # 3. Generate the detailed prompt for Gemini
                    meta_prompt = report_generator.generate_meta_prompt(patient_data, prediction_json)

                    # 4. Get the final, user-friendly report from Gemini
                    llm_report = report_generator.get_llm_report(meta_prompt)

                    # 5. Check for errors from the API call
                    if llm_report.startswith("ERROR:"):
                        print(f"\nSkipping row {index} due to an API error: {llm_report}")
                        continue

                    # 6. Create the user-centric input prompt for the fine-tuning dataset
                    user_centric_prompt = (
                        "Please provide a detailed, easy-to-understand health report based on the "
                        "following clinical data and model analysis:\n\n"
                        f"{prediction_json}"
                    )
                    
                    # 7. Create the final JSON object and write it to the file
                    data_record = {
                        "input": user_centric_prompt,
                        "output": llm_report
                    }
                    f.write(json.dumps(data_record) + '\n')

                except Exception as e:
                    print(f"\nAn unexpected error occurred at index {index}: {e}. Skipping row.")
                    continue
        
        print(f"\nPipeline complete. Dataset saved to {OUTPUT_FILE}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure you have run the 'train_and_save_artifacts.py' script first to generate the necessary files.")
    except Exception as e:
        print(f"A critical error occurred: {e}")
