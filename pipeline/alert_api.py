"""
Legacy shim (kept for backward compatibility).

New canonical API lives in `src.api.main` and includes:
- POST /predict
- Legacy POST /score, /score/features
"""

from src.api.main import app  # re-export FastAPI app
