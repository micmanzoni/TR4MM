from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strategy import scan_market, MIN_SCORE_DEFAULT

app = FastAPI(title="Michiman Trading Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/scan")
def scan(min_score: float = MIN_SCORE_DEFAULT):
    """
    Calcola l'analisi al volo usando scan_market().
    Nessun file results.json, niente dipendenze extra.
    """
    data = scan_market(min_score=min_score)
    return data

