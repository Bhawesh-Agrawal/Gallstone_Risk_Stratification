import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

DATASET = os.path.join(DATASET_DIR, "gallstone_.csv")
RESULT_DIR = os.path.join(BASE_DIR, "models")



print(BASE_DIR)