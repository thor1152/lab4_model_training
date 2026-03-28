import sys, os

# Ensure src is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.bc_data import generate_data_bc

if __name__ == "__main__":
    generate_data("data/breast_cancer.csv")
