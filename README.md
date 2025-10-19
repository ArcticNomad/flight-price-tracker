Question 1 — System Design

--Goal--

To design an automated system that tracks flight ticket prices for different routes, airlines, and flight dates over regular time intervals (e.g., every 7 days or 15 days).

The system should automatically record ticket prices over time and generate a time-series view of price fluctuations leading up to the flight.

--System Overview--

Input Parameters:

 Route (e.g., LHE → BKK)

 Airline (e.g., “Thai Airways”)

 Flight Date (e.g., 2026-01-01)

 Tracking Interval (e.g., 15 days)

Output Example

15 Oct 2025 – $670  
1 Nov 2025 – $678  
15 Nov 2025 – $712

--System Flow--

User Input – User provides flight details (route, airline, flight date, interval).

Data Storage – System stores these details in MongoDB.

Scheduler Activation – When tracking start date arrives, a background job runs every X days to fetch the current price (from an API or dummy data).

Price Logging – Each new price is appended to that flight’s record in a prices[] array.

Data Access – User can query the API to see the time-series of prices or analyze trend


--Database Schema Design (MongoDB)--

| Field            | Type             | Description                             |
| :--------------- | :--------------- | :-------------------------------------- |
| `_id`            | ObjectId         | Unique flight record                    |
| `route`          | String           | e.g., `LHE → BKK`                       |
| `airline`        | String           | Airline name                            |
| `flightDate`     | Date             | Actual flight date                      |
| `trackStartDate` | Date             | When to begin tracking                  |
| `intervalDays`   | Number           | Interval (in days) between price checks |
| `prices`         | Array of Objects | Time-series entries `{date, price}`     |

