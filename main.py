from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strategy import scan_market

app = FastAPI(title="Trading Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # per cominciare va bene così
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/scan")
def scan(min_score: float = 0.01):
    """
    Endpoint principale:
    GET /scan?min_score=0.01
    Restituisce JSON con:
    - market: stato generale del mercato
    - assets: lista di titoli/ETF con entry/SL/TP
    """
    data = scan_market(min_score=min_score)
    return data
