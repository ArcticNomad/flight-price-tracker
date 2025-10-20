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
        fieldnames=[
            "_id",
            "route",
            "route_desc",
            "flight_date",
            "airline",
            "price_date",
            "price",
            "description",
            "interval",
            "threshold_months"
        ]
    )
    writer.writeheader()

    for flight in flights:
        flight_id = str(flight["_id"])
        route = flight.get("route", "")
        route_desc = flight.get("route_desc", "")
        flight_date = flight.get("flight_date", "")
        airline = flight.get("airline", "")
        description = flight.get("description", "")
        interval = flight.get("interval", "")
        threshold = flight.get("threshold_months", "")

        prices = flight.get("prices", [])
        if prices:
            for price_entry in prices:
                writer.writerow({
                    "_id": flight_id,
                    "route": route,
                    "route_desc": route_desc,
                    "flight_date": flight_date,
                    "airline": airline,
                    "price_date": price_entry.get("date", ""),
                    "price": price_entry.get("price", ""),
                    "description": description,
                    "interval": interval,
                    "threshold_months": threshold
                })
        else:
            # Write one row even if no price data exists
            writer.writerow({
                "_id": flight_id,
                "route": route,
                "route_desc": route_desc,
                "flight_date": flight_date,
                "airline": airline,
                "price_date": "",
                "price": "",
                "description": description,
                "interval": interval,
                "threshold_months": threshold
            })

print(f"✅ CSV file '{output_file}' created successfully.")
print(f"📦 Total flights exported: {collection.count_documents({})}")
