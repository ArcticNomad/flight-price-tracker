from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import random


client = MongoClient("mongodb://localhost:27017/")
db = client["flight_tracker"]
collection = db["flights"]
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


def simulate_price() -> str:
    return f"{random.randint(500, 1000)}$"


def generate_price_history(flight_date: str, interval: str, threshold_months: int) -> list:
    flight_date_dt = datetime.fromisoformat(flight_date)
    start_date = flight_date_dt - timedelta(days=threshold_months * 30)
    current_date = datetime.now()
    
    if start_date > current_date:
        return []
    end_date = min(current_date, flight_date_dt)

    interval_days = {"daily": 1, "weekly": 7, "biweekly": 15}
    days = interval_days.get(interval, 15)

    prices = []
    current = start_date
    while current <= end_date:
        prices.append({
            "date": current.strftime("%Y-%m-%d"),
            "price": simulate_price()
        })
        current += timedelta(days=days)
    return prices


def seed_flights():
    collection.delete_many({})
    sample_flights = [
        {"route": "LHE→BKK", "flight_date": "2026-01-01", "airline": "Thai Airways", "interval": "biweekly", "threshold_months": 6},
        {"route": "SIN→JED", "flight_date": "2026-02-15", "airline": "Singapore Airlines", "interval": "weekly", "threshold_months": 4},
        {"route": "DXB→LHR", "flight_date": "2026-03-10", "airline": "Emirates", "interval": "biweekly", "threshold_months": 6},
        {"route": "JFK→CDG", "flight_date": "2026-04-20", "airline": "Air France", "interval": "daily", "threshold_months": 3},
        {"route": "SYD→LAX", "flight_date": "2026-05-05", "airline": "Qantas", "interval": "weekly", "threshold_months": 5},
        {"route": "DEL→SFO", "flight_date": "2026-06-12", "airline": "Air India", "interval": "biweekly", "threshold_months": 6},
        {"route": "HKG→NRT", "flight_date": "2026-07-01", "airline": "Cathay Pacific", "interval": "weekly", "threshold_months": 4},
        {"route": "AMS→KUL", "flight_date": "2026-08-15", "airline": "KLM", "interval": "biweekly", "threshold_months": 6},
        {"route": "BKK→CDG", "flight_date": "2026-09-20", "airline": "Lufthansa", "interval": "daily", "threshold_months": 3},
        {"route": "MIA→GRU", "flight_date": "2026-10-10", "airline": "American Airlines", "interval": "weekly", "threshold_months": 5}
    ]

    for flight in sample_flights:
        description = f"Flight from {flight['route']} on {flight['flight_date']} with {flight['airline']}"
        embedding = model.encode([description])[0].tolist()
        prices = generate_price_history(flight["flight_date"], flight["interval"], flight["threshold_months"])
        collection.insert_one({
            "route": flight["route"],
            "flight_date": flight["flight_date"],
            "airline": flight["airline"],
            "prices": prices,
            "description": description,
            "embedding": embedding,
            "interval": flight["interval"],
            "threshold_months": flight["threshold_months"]
        })
    print(f"Seeded {len(sample_flights)} flights into MongoDB")

if __name__ == "__main__":
    seed_flights()