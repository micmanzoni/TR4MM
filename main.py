from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strategy import scan_market, MIN_SCORE_DEFAULT

import traceback
import io

app = FastAPI(title="Michiman Trading Scanner (DEBUG)")

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
    VERSIONE DEBUG:
    prova a eseguire scan_market();
    se qualcosa va storto, invece di 500 ritorna l'errore in chiaro.
    """
    try:
        data = scan_market(min_score=min_score)
        return data
    except Exception as e:
        buf = io.StringIO()
        traceback.print_exc(file=buf)
        return {
            "error": str(e),
            "traceback": buf.getvalue(),
        }
