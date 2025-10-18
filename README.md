-Overview-

The Flight Price Tracker helps users monitor ticket price fluctuations for specific flight routes and dates.
You can define:

The flight route (e.g. LHE → BKK)

The airline

The departure date

The tracking interval (e.g. 7 days or 15 days)

The system periodically collects and stores ticket prices, building a time-series dataset of prices leading up to the flight.

⚙️ Features

🛫 Track ticket prices for any route and airline

📅 Time-series data tracking (weekly, biweekly, or custom intervals)

💾 MongoDB storage for all flight and price data

🔍 Search flights using hybrid search (text + numeric filters)

🧮 Optional hybrid ranking with weighted formula

🧰 RESTful APIs for easy integration and visualization


🧱 Tech Stack
Component	Technology
Backend	Node.js + Express
Database	MongoDB
Search	MongoDB Atlas Search / Hybrid text search
API Testing	Postman
Data	Seeded JSON dataset (flights.json
