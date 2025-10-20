from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
import uvicorn
from enum import Enum
from bson import ObjectId
from datetime import datetime, timedelta

app = FastAPI()
client = MongoClient("mongodb://localhost:27017/")
db = client["flight_tracker"]
collection = db["flights"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

class IntervalType(str, Enum):
    daily = "daily"
    weekly = "weekly"
    biweekly = "biweekly"

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
    prices: list[PriceEntry] | None = None
    description: str
    interval: str
    threshold_months: int
    route_desc: str | None = None

def generate_embedding(text: str) -> list:
    return model.encode([text])[0].tolist()

def normalize(arr):
    arr = np.array(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

def filter_prices(prices: list, interval: str, threshold_months: int, flight_date: str) -> list:
    if not prices:
        print(f"No prices available to filter for {flight_date}, interval={interval}")
        return []
    
    flight_date_dt = datetime.fromisoformat(flight_date)
    start_date = flight_date_dt - timedelta(days=threshold_months * 30)
    interval_days = {"daily": 1, "weekly": 7, "biweekly": 15}
    days = interval_days.get(interval, 15)
    
    filtered_prices = []
    prices = sorted(prices, key=lambda x: x["date"])  
    current = start_date
    
    for price in prices:
        price_date = datetime.fromisoformat(price["date"])
        if price_date >= start_date and price_date <= flight_date_dt:
            if price_date >= current:
                filtered_prices.append(price)
                current += timedelta(days=days)
    
    print(f"Filtered {len(prices)} daily prices to {len(filtered_prices)} {interval} prices for {flight_date}")
    return filtered_prices

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
                "path": "/track-prices",
                "method": "GET",
                "description": "Tracks a flight and filters prices by interval.",
                "parameters": {
                    "route": "str",
                    "flight_date": "str (ISO format)",
                    "airline": "str",
                    "interval": "str (daily, weekly, biweekly)",
                    "threshold_months": "int"
                },
                "example_request": "curl 'http://localhost:8000/track-prices?route=DEL%E2%86%92SFO&flight_date=2026-06-12&airline=Air%20India&interval=weekly&threshold_months=6'",
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

@app.get("/favicon.ico")
async def favicon():
    return {"status": "no favicon"}

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
        "airline": airline
    })
    print(f"Queried flight: {route}, {flight_date}, {airline}, interval={interval}")
    if existing_flight:
        filtered_prices = filter_prices(existing_flight.get("prices", []), interval, threshold_months, flight_date)
        existing_flight["_id"] = str(existing_flight["_id"])
        flight_response = {k: v for k, v in existing_flight.items() if k != "embedding"}
        flight_response["prices"] = filtered_prices
        flight_response["interval"] = interval
        flight_response["threshold_months"] = threshold_months
        print(f"Returning flight with {len(filtered_prices)} prices")
        return flight_response

    description = f"Flight from {route} on {flight_date} with {airline}"
    embedding = generate_embedding(description)

    flight_data = {
        "route": route,
        "flight_date": flight_date,
        "airline": airline,
        "description": description,
        "embedding": embedding,
        "threshold_months": threshold_months
    }
    result = collection.insert_one(flight_data)
    flight_response = {
        "_id": str(result.inserted_id),
        "route": route,
        "flight_date": flight_date,
        "airline": airline,
        "prices": None,
        "description": description,
        "interval": interval,
        "threshold_months": threshold_months
    }
    print(f"No flight found, created new flight with ID {flight_response['_id']}")
    return flight_response

@app.get("/flights/search")
async def search_flights(query: str):
    docs = list(collection.find({}))
    if not docs:
        return {"results": []}

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

    query_embedding = generate_embedding(query)
    doc_embeddings = np.array([d["embedding"] for d in docs])
    semantic_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    tokenized_corpus = []
    match_scores = []
    for doc in docs:
        description_tokens = word_tokenize(doc["description"].lower())
        route_tokens = doc["route"].lower().replace("→", " ").split()
        origin_token = route_tokens[0]
        dest_token = route_tokens[1]
        corpus = description_tokens + [origin_token] * 10 + [dest_token]
        tokenized_corpus.append(corpus)
        match_score = 1.0
        if origin and destination:
            origin_match = origin in doc["route"].lower() or any(o in doc["route"].lower() for o in country_to_airports.get(origin, []))
            dest_match = destination in doc["route"].lower() or any(d in doc["route"].lower() for d in country_to_airports.get(destination, []))
            if not (origin_match and dest_match):
                match_score = 0.3
        match_scores.append(match_score)
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    print(f"Query: {query}")
    print(f"Tokenized query: {tokenized_query}")
    for i, doc in enumerate(docs):
        print(f"Flight {doc['route']} ({doc['_id']}): Semantic={semantic_similarities[i]:.4f}, BM25={bm25_scores[i]:.4f}, Match={match_scores[i]:.4f}")

    alpha = 0.7
    semantic_scaled = semantic_similarities / np.max(semantic_similarities) if np.max(semantic_similarities) > 0 else np.ones_like(semantic_similarities) * 0.5
    bm25_scaled = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else np.ones_like(bm25_scores) * 0.5
    hybrid_scores = (alpha * semantic_scaled + (1 - alpha) * bm25_scaled) * match_scores

    top_indices = hybrid_scores.argsort()[-5:][::-1]
    results = []
    for idx in top_indices:
        if hybrid_scores[idx] < 0.15:
            continue
        doc = docs[idx]
        doc["_id"] = str(doc["_id"])
        doc.pop("embedding", None)
        results.append({
            "flight": doc,
            "hybrid_score": round(float(hybrid_scores[idx]), 4)
        })

    for idx in top_indices:
        print(f"Top result: {docs[idx]['route']} ({docs[idx]['_id']}): Hybrid={hybrid_scores[idx]:.4f}")
    return {"results": results}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)