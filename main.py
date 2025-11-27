from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strategy import scan_market
import json
import os

app = FastAPI(title="Michiman Trading Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_FILE = "results.json"

def load_results_from_file():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@app.get("/scan")
def scan(min_score: float = 0.01):
    """
    Se esiste results.json, restituisce l'ultima analisi salvata.
    Altrimenti calcola al volo scan_market().
    """
    data = load_results_from_file()
    if data is None:
        data = scan_market(min_score=min_score)
    return data
