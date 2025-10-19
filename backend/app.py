import random
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
import uvicorn
import asyncio
from enum import Enum
from bson import ObjectId
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Initialize FastAPI and MongoDB
app = FastAPI()
client = MongoClient("mongodb://localhost:27017/")
db = client["flight_tracker"]
collection = db["flights"]

# Initialize SentenceTransformer with CPU explicitly
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

# Scheduler for price updates
scheduler = AsyncIOScheduler()

# Enum for interval types
class IntervalType(str, Enum):
    daily = "daily"
    weekly = "weekly"
    biweekly = "biweekly"

# Pydantic models for input validation
class FlightInput(BaseModel):
    route: str
    flight_date: str
    airline: str
    interval: IntervalType = IntervalType.biweekly
    threshold_months: int = 6

class PriceEntry(BaseModel):
    date: str
    price: str

class FlightResponse(BaseModel):
    route: str
    flight_date: str
    airline: str
    prices: list[PriceEntry]
    description: str
    interval: str
    threshold_months: int

# Helper functions
def generate_embedding(text: str) -> list:
    return model.encode([text])[0].tolist()

def normalize(arr):
    arr = np.array(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

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

# Lifespan handler for scheduler
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(update_prices, 'interval', minutes=1)
    scheduler.start()
    print("Scheduler started")
    try:
        yield
    finally:
        scheduler.shutdown()
        print("Scheduler shutdown")

app = FastAPI(lifespan=lifespan)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Flight Price Tracker API",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Returns a welcome message and list of available endpoints.",
                "parameters": "None",
                "example_request": "curl http://localhost:8000/",
            },
           
            {
                "path": "/flights/seed",
                "method": "POST",
                "description": "Seeds 10 sample flights.",
                "parameters": "None",
                "example_request": "curl -X POST http://localhost:8000/flights/seed",

            },
            {
                "path": "/track-prices",
                "method": "GET",
                "description": "Tracks prices for a flight.",
                "parameters": {
                    "route": "str",
                    "flight_date": "str (ISO format)",
                    "airline": "str",
                    "interval": "str (daily, weekly, biweekly)",
                    "threshold_months": "int"
                },
                "example_request": "curl 'http://localhost:8000/track-prices?route=LHE→BKK&flight_date=2026-01-01&airline=Thai%20Airways&interval=daily&threshold_months=6'",
            },
            {
                "path": "/flights",
                "method": "GET",
                "description": "Lists all flights.",
                "parameters": "None",
                "example_request": "curl http://localhost:8000/flights",
              
            },
            {
                "path": "/flights/{flight_id}",
                "method": "GET",
                "description": "Retrieves a flight by ID.",
                "parameters": {"flight_id": "str (MongoDB ObjectId)"},
                "example_request": "curl http://localhost:8000/flights/6713f...",
            
            },
            {
                "path": "/flights/search",
                "method": "GET",
                "description": "Hybrid search for flights.",
                "parameters": {"query": "str"},
                "example_request": "curl 'http://localhost:8000/flights/search?query=flight%20from%20Bangkok%20to%20France'",
            
            }
        ]
    }

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    return {"status": "no favicon"}

# Seeding function
@app.post("/flights/seed")
async def seed_flights():
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
        {"route": "BKK→CDG", "flight_date": "2026-09-20", "airline": "Lufthansa", "interval": "daily", "threshold_months": 3},  # Replaced BKK→FRA
        {"route": "MIA→GRU", "flight_date": "2026-10-10", "airline": "American Airlines", "interval": "weekly", "threshold_months": 5}
    ]

    for flight in sample_flights:
        description = f"Flight from {flight['route']} on {flight['flight_date']} with {flight['airline']}"
        embedding = generate_embedding(description)
        prices = generate_price_history(flight["flight_date"], flight["interval"], flight["threshold_months"])
        collection.insert_one({
            "route": flight["route"],
            "flight_date": "2026-01-01",
            "airline": flight["airline"],
            "prices": prices,
            "description": description,
            "embedding": embedding,
            "interval": flight["interval"],
            "threshold_months": flight["threshold_months"]
        })
    return {"message": f"Seeded {len(sample_flights)} flights into MongoDB"}

# Track prices endpoint
@app.get("/track-prices", response_model=FlightResponse)
async def track_prices(route: str, flight_date: str, airline: str, interval: IntervalType = IntervalType.biweekly, threshold_months: int = 6):
    try:
        flight_date_dt = datetime.fromisoformat(flight_date)
        if flight_date_dt < datetime.now():
            raise HTTPException(status_code=400, detail="Flight date must be in the future")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid flight date format")

    existing_flight = collection.find_one({
        "route": route,
        "flight_date": flight_date,
        "airline": airline,
        "interval": interval
    })
    if existing_flight:
        existing_flight["_id"] = str(existing_flight["_id"])
        return {k: v for k, v in existing_flight.items() if k != "embedding"}

    description = f"Flight from {route} on {flight_date} with {airline}"
    embedding = generate_embedding(description)
    prices = generate_price_history(flight_date, interval, threshold_months)

    flight_data = {
        "route": route,
        "flight_date": flight_date,
        "airline": airline,
        "prices": prices,
        "description": description,
        "embedding": embedding,
        "interval": interval,
        "threshold_months": threshold_months
    }
    result = collection.insert_one(flight_data)
    flight_response = {
        "_id": str(result.inserted_id),
        "route": route,
        "flight_date": flight_date,
        "airline": airline,
        "prices": prices,
        "description": description,
        "interval": interval,
        "threshold_months": threshold_months
    }
    return flight_response

# Search endpoint
@app.get("/flights/search")
async def search_flights(query: str):
    """Hybrid search for flights by route, airline, or description."""
    docs = list(collection.find({}))
    if not docs:
        return {"results": []}

    # Map country to airport codes
    country_to_airports = {
        "pakistan": ["LHE", "ISB", "KHI", "PEW"],
        "france": ["CDG", "ORY", "NCE"],
        "germany": ["FRA", "MUC"],
        "saudi arabia": ["JED", "RUH"],
        "uk": ["LHR", "LGW"],
        "usa": ["JFK", "LAX", "SFO", "MIA"],
        "japan": ["NRT", "HND"],
        "australia": ["SYD"],
        "singapore": ["SIN"],
        "malaysia": ["KUL"],
        "thailand": ["BKK"],
        "brazil": ["GRU"],
        "india": ["DEL"],
        "hong kong": ["HKG"],
        "uae": ["DXB"],
        "netherlands": ["AMS"]
    }

    # Preprocess query: correct typos and add airport codes
    query_lower = query.lower().replace("honkong", "hong kong")
    origin, destination = None, None
    if "from" in query_lower and "to" in query_lower:
        parts = query_lower.split("from")[1].split("to")
        if len(parts) == 2:
            origin = parts[0].strip()
            destination = parts[1].strip()
    for country, airports in country_to_airports.items():
        if country in query_lower:
            query_lower += " " + " ".join(airports)
    tokenized_query = word_tokenize(query_lower)

    # Semantic search
    query_embedding = generate_embedding(query)
    doc_embeddings = np.array([d["embedding"] for d in docs])
    semantic_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Keyword search with BM25
    tokenized_corpus = []
    match_scores = []
    for doc in docs:
        description_tokens = word_tokenize(doc["description"].lower())
        route_tokens = doc["route"].lower().replace("→", " ").split()
        origin_token = route_tokens[0]
        dest_token = route_tokens[1]
        # Boost origin significantly
        corpus = description_tokens + [origin_token] * 10 + [dest_token]
        tokenized_corpus.append(corpus)
        # Check if route matches query's origin and destination
        match_score = 1.0
        if origin and destination:
            # Check origin match (query origin or its airports)
            origin_match = origin in doc["route"].lower() or any(o in doc["route"].lower() for o in country_to_airports.get(origin, []))
            # Check destination match (query destination or its airports)
            dest_match = destination in doc["route"].lower() or any(d in doc["route"].lower() for d in country_to_airports.get(destination, []))
            if not (origin_match and dest_match):
                match_score = 0.3  # Strong penalty for non-matching routes
        match_scores.append(match_score)
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Debug: Log raw scores
    print(f"Query: {query}")
    print(f"Tokenized query: {tokenized_query}")
    for i, doc in enumerate(docs):
        print(f"Flight {doc['route']} ({doc['_id']}): Semantic={semantic_similarities[i]:.4f}, BM25={bm25_scores[i]:.4f}, Match={match_scores[i]:.4f}")

    # Hybrid scoring with match penalty
    alpha = 0.7
    semantic_scaled = semantic_similarities / np.max(semantic_similarities) if np.max(semantic_similarities) > 0 else np.ones_like(semantic_similarities) * 0.5
    bm25_scaled = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else np.ones_like(bm25_scores) * 0.5
    hybrid_scores = (alpha * semantic_scaled + (1 - alpha) * bm25_scaled) * match_scores

    # Get top 5 results
    top_indices = hybrid_scores.argsort()[-5:][::-1]
    results = []
    for idx in top_indices:
        if hybrid_scores[idx] < 0.15:  # Lower threshold to include valid results
            continue
        doc = docs[idx]
        doc["_id"] = str(doc["_id"])
        doc.pop("embedding", None)
        results.append({
            "flight": doc,
            "hybrid_score": round(float(hybrid_scores[idx]), 4)
        })

    # Debug: Log final scores
    for idx in top_indices:
        print(f"Top result: {docs[idx]['route']} ({docs[idx]['_id']}): Hybrid={hybrid_scores[idx]:.4f}")
    return {"results": results}
# Flight ID endpoint
@app.get("/flights/{flight_id}", response_model=FlightResponse)
async def get_flight(flight_id: str):
    try:
        flight = collection.find_one({"_id": ObjectId(flight_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid flight ID format")
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")
    flight["_id"] = str(flight["_id"])
    return {k: v for k, v in flight.items() if k != "embedding"}

@app.get("/flights", response_model=dict)
async def list_flights():
    flights = list(collection.find({}))
    for flight in flights:
        flight["_id"] = str(flight["_id"])
        flight.pop("embedding", None)
    return {"flights": flights}

# Scheduler for price updates
async def update_prices():
    flights = collection.find({})
    current_date = datetime.now()
    for flight in flights:
        if "threshold_months" not in flight:
            print(f"Skipping flight {flight.get('route', 'unknown')} due to missing threshold_months")
            continue
        flight_date = datetime.fromisoformat(flight["flight_date"])
        threshold_date = flight_date - timedelta(days=flight["threshold_months"] * 30)
        if current_date >= threshold_date and current_date < flight_date:
            interval_days = {"daily": 1, "weekly": 7, "biweekly": 15}
            days = interval_days.get(flight["interval"], 15)
            last_price_date = datetime.fromisoformat(flight["prices"][-1]["date"]) if flight["prices"] else threshold_date
            if (current_date - last_price_date).days >= days:
                new_price = {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "price": simulate_price()
                }
                collection.update_one(
                    {"_id": flight["_id"]},
                    {"$push": {"prices": new_price}}
                )
                print(f"Updated price for flight {flight['route']} on {flight['flight_date']} with interval {flight['interval']}")

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)