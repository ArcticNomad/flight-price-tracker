import csv
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["flight_tracker"]
collection = db["flights"]

# Retrieve all flights, excluding embedding
flights = collection.find({}, {"embedding": 0})

# Prepare CSV file
output_file = "seeded_flights_prices_expanded.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["_id", "route", "flight_date", "airline", "price_date", "price", "description", "interval", "threshold_months"]
    )
    writer.writeheader()

    for flight in flights:
        flight_id = str(flight["_id"])
        for price_entry in flight.get("prices", []):
            writer.writerow({
                "_id": flight_id,
                "route": flight["route"],
                "flight_date": flight["flight_date"],
                "airline": flight["airline"],
                "price_date": price_entry["date"],
                "price": price_entry["price"],
                "description": flight["description"],
                "interval": flight.get("interval", ""),
                "threshold_months": flight.get("threshold_months", "")
            })
        if not flight.get("prices"):
            writer.writerow({
                "_id": flight_id,
                "route": flight["route"],
                "flight_date": flight["flight_date"],
                "airline": flight["airline"],
                "price_date": "",
                "price": "",
                "description": flight["description"],
                "interval": flight.get("interval", ""),
                "threshold_months": flight.get("threshold_months", "")
            })

print(f"CSV file '{output_file}' created successfully.")
print(f"Total flights exported: {collection.count_documents({})}")