from pathlib import Path

# Define project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
TRAIN_PATH = BASE_DIR / 'data' / 'train.csv'
TEST_PATH = BASE_DIR / 'data' / 'test.csv'

# Submission paths
SUBMISSION_DIR = BASE_DIR / 'submission'
SUBMISSION_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

TARGET = 'SalePrice'