# scripts/serve_api.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from app.bc_api import create_app_bc
app = create_app_bc("models/breast_cancer_model.pkl")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)