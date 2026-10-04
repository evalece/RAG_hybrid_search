import joblib

data = [
    {
        "place": "Banff National Park",
        "state": "Alberta",
        "description": "A mountain destination known for turquoise lakes, hiking trails, and dramatic Rocky Mountain scenery.",
        "best_season_to_visit": "Summer",
        "attractions": "Lake Louise, Moraine Lake, Banff Gondola",
        "budget": "High",
        "user_ratings": 4.8,
        "last_updated": "2026-10-04T00:00:00Z"
    },
    {
        "place": "Toronto",
        "state": "Ontario",
        "description": "A large Canadian city known for diverse food, museums, waterfront areas, shopping, and urban attractions.",
        "best_season_to_visit": "Spring and Fall",
        "attractions": "CN Tower, Royal Ontario Museum, Toronto Islands",
        "budget": "Medium to High",
        "user_ratings": 4.5,
        "last_updated": "2026-10-04T00:00:00Z"
    },
    {
        "place": "Montreal",
        "state": "Quebec",
        "description": "A historic and lively city known for French Canadian culture, architecture, restaurants, festivals, and nightlife.",
        "best_season_to_visit": "Summer and Fall",
        "attractions": "Old Montreal, Mount Royal, Notre-Dame Basilica",
        "budget": "Medium",
        "user_ratings": 4.6,
        "last_updated": "2026-10-04T00:00:00Z"
    }
]
joblib.dump(data, "data.joblib")

print("Saved", len(data), "places to data.joblib")