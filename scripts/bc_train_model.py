import sys, os

# Ensure src is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.bc_data import load_data_bc
from ml_pipeline.bc_model import train_model_bc

if __name__ == "__main__":
    df = load_data_bc("data/breast_cancer.csv")
    train_model_bc(df, "models/breast_cancer_model.pkl")