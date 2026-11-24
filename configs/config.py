from pathlib import Path

# get main directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
HARMONIZED_DATA_PATH = DATA_DIR / "harmonized_data_2292026_rightSMILES.parquet"