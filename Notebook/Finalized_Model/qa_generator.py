import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
from tqdm import tqdm
import re

class QAGenerator:
    """
    A class to generate a question-and-answer dataset about gallstones using the Gemini API.
    """
    def __init__(self, api_key):
        """
        Initializes the generator and configures the Gemini API.
        """
        if not api_key:
            raise ValueError("API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        print("Gemini API configured successfully for Q&A generation.")

    def generate_question_list(self, num_questions=300):
        """
        Uses the LLM to generate a diverse list of questions about gallstones.
        """
        print(f"Generating a list of {num_questions} unique questions... This may take a moment.")
        prompt = (
            f"Please generate a numbered list of {num_questions} unique and diverse questions about gallstones. "
            "Cover a wide range of topics including: causes, symptoms (common and rare), risk factors, "
            "diagnosis methods (ultrasound, CT scan, HIDA scan), different types of gallstones, "
            "treatment options (surgery, medication, ERCP), prevention strategies, diet and nutrition, "
            "complications (pancreatitis, cholecystitis), gallstones in different populations (children, pregnant women), "
            "and recovery after surgery. Ensure the questions are distinct from each other."
        )
        try:
            response = self.model.generate_content(prompt)
            # Use regex to find all numbered list items, which is more robust
            questions = re.findall(r'^\d+\.\s*(.*)', response.text, re.MULTILINE)
            if len(questions) < num_questions:
                print(f"Warning: LLM generated only {len(questions)} questions.")
            return questions
        except Exception as e:
            print(f"Failed to generate questions: {e}")
            return []

    def get_answer_for_question(self, question):
        """
        Uses the LLM to generate a detailed answer for a single question.
        """
        prompt = (
            "You are a medical expert providing clear, accurate, and easy-to-understand information. "
            "Please provide a detailed and comprehensive answer to the following question about gallstones:\n\n"
            f"Question: \"{question}\"\n\n"
            "Answer:"
        )
        try:
            # Add a delay to respect API rate limits
            time.sleep(1.5)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"ERROR: An error occurred while communicating with the Gemini API: {e}"

if __name__ == "__main__":
    load_dotenv()

    # --- Configuration ---
    NUM_QUESTIONS_TO_GENERATE = 300
    QUESTIONS_FILE = "gallstone_questions.json"
    OUTPUT_FILE = "gallstone_qa_dataset.jsonl"

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in the .env file.")

        qa_generator = QAGenerator(api_key=api_key)

        # --- Step 1: Generate or Load Questions ---
        questions = []
        if os.path.exists(QUESTIONS_FILE):
            print(f"Loading questions from existing file: {QUESTIONS_FILE}")
            with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        else:
            questions = qa_generator.generate_question_list(NUM_QUESTIONS_TO_GENERATE)
            if questions:
                with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, indent=2)
                print(f"Successfully generated and saved {len(questions)} questions to {QUESTIONS_FILE}.")
            else:
                raise Exception("Could not generate the initial list of questions. Aborting.")

        # --- Step 2: Generate Answers (Resumable) ---
        start_index = 0
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                start_index = sum(1 for _ in f)
            print(f"Resuming from record {start_index}. Found {start_index} existing Q&A pairs.")

        print(f"Total questions to process: {len(questions)}. Starting from index: {start_index}.")

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            # Use tqdm for a progress bar, starting from the correct index
            for i in tqdm(range(start_index, len(questions)), initial=start_index, total=len(questions), desc="Generating Q&A Pairs"):
                question = questions[i]
                
                # Get the answer from the LLM
                answer = qa_generator.get_answer_for_question(question)

                if answer.startswith("ERROR:"):
                    print(f"\nAPI Error for question {i+1}: {answer}. Stopping for now. You can restart the script later.")
                    # Break the loop on API error, can be restarted later
                    break
                
                # Format for fine-tuning dataset
                data_record = {
                    "input": question,
                    "output": answer
                }
                f.write(json.dumps(data_record) + '\n')

        print(f"\nQ&A generation pipeline complete. Dataset saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"A critical error occurred: {e}")
