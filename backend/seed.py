from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import random

client = MongoClient("mongodb://localhost:27017/")
db = client["flight_tracker"]
collection = db["flights"]

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Generate random ticket price
def simulate_price() -> str:
    return f"{random.randint(500, 1000)}$"

# Generate daily price history between threshold and flight date
def generate_price_history(flight_date: str, threshold_months: int) -> list:
    flight_date_dt = datetime.fromisoformat(flight_date)
    start_date = flight_date_dt - timedelta(days=threshold_months * 30)
    end_date = flight_date_dt

    prices = []
    current = start_date
    while current <= end_date:
        prices.append({
            "date": current.strftime("%Y-%m-%d"),
            "price": simulate_price()
        })
        current += timedelta(days=1)
    print(f"Generated {len(prices)} prices for flight_date={flight_date} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    return prices

# Seed the database
def seed_flights():
    collection.delete_many({})  
    sample_flights = [
        {"route": "LHE→BKK", "route_desc": "Lahore → Bangkok", "flight_date": "2026-01-01", "airline": "Thai Airways", "threshold_months": 6},
        {"route": "SIN→JED", "route_desc": "Singapore → Jeddah", "flight_date": "2026-02-15", "airline": "Singapore Airlines", "threshold_months": 4},
        {"route": "DXB→LHR", "route_desc": "Dubai → London Heathrow", "flight_date": "2026-03-10", "airline": "Emirates", "threshold_months": 6},
        {"route": "JFK→CDG", "route_desc": "New York (JFK) → Paris (Charles de Gaulle)", "flight_date": "2026-04-20", "airline": "Air France", "threshold_months": 3},
        {"route": "SYD→LAX", "route_desc": "Sydney → Los Angeles", "flight_date": "2026-05-05", "airline": "Qantas", "threshold_months": 5},
        {"route": "DEL→SFO", "route_desc": "Delhi → San Francisco", "flight_date": "2026-06-12", "airline": "Air India", "threshold_months": 6},
        {"route": "HKG→NRT", "route_desc": "Hong Kong → Tokyo Narita", "flight_date": "2026-07-01", "airline": "Cathay Pacific", "threshold_months": 4},
        {"route": "AMS→KUL", "route_desc": "Amsterdam → Kuala Lumpur", "flight_date": "2026-08-15", "airline": "KLM", "threshold_months": 6},
        {"route": "BKK→CDG", "route_desc": "Bangkok → Paris (Charles de Gaulle)", "flight_date": "2026-09-20", "airline": "Lufthansa", "threshold_months": 3},
        {"route": "MIA→GRU", "route_desc": "Miami → São Paulo (Guarulhos)", "flight_date": "2026-10-10", "airline": "American Airlines", "threshold_months": 5}
    ]

    for flight in sample_flights:
        description = f"Flight from {flight['route_desc']} on {flight['flight_date']} with {flight['airline']}"
        embedding = model.encode([description])[0].tolist()
        prices = generate_price_history(flight["flight_date"], flight["threshold_months"])
        
        collection.insert_one({
            "route": flight["route"],
            "route_desc": flight["route_desc"],
            "flight_date": flight["flight_date"],
            "airline": flight["airline"],
            "prices": prices,
            "description": description,
            "embedding": embedding,
            "threshold_months": flight["threshold_months"]
        })
        print(f"Seeded flight: {flight['route']} on {flight['flight_date']} with {len(prices)} prices")
    
    print(f"Seeded {len(sample_flights)} flights into MongoDB")

if __name__ == "__main__":
    seed_flights()