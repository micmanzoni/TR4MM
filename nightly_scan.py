import json
from strategy import scan_market

def main():
    print("Eseguo scan_market() per aggiornare results.json...")
    data = scan_market()
    with open("results.json", "w") as f:
        json.dump(data, f)
    print("results.json aggiornato.")

if __name__ == "__main__":
    main()
