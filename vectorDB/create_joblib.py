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
    },

    # ----- WINTER -----

    {
        "place": "Quebec City",
        "state": "Quebec",
        "description": "A historic city especially popular in winter for snowy streets, festivals, outdoor activities, and European-style architecture.",
        "best_season_to_visit": "Winter",
        "attractions": "Old Quebec, Quebec Winter Carnival, Terrasse Dufferin",
        "budget": "Low",
        "user_ratings": 4.7,
        "last_updated": "2026-10-04T00:00:00Z"
    },
    {
        "place": "Whistler",
        "state": "British Columbia",
        "description": "A famous mountain resort offering skiing, snowboarding, luxury accommodation, and alpine scenery.",
        "best_season_to_visit": "Winter",
        "attractions": "Whistler Blackcomb, Peak 2 Peak Gondola, Whistler Village",
        "budget": "High",
        "user_ratings": 4.8,
        "last_updated": "2026-10-04T00:00:00Z"
    },
    {
        "place": "Ottawa",
        "state": "Ontario",
        "description": "Canada's capital offers museums, historic landmarks, winter skating, and relatively affordable sightseeing.",
        "best_season_to_visit": "Winter",
        "attractions": "Rideau Canal, Parliament Hill, Canadian Museum of History",
        "budget": "Medium",
        "user_ratings": 4.4,
        "last_updated": "2026-10-04T00:00:00Z"
    },

    # ----- SPRING -----

    {
        "place": "Victoria",
        "state": "British Columbia",
        "description": "A coastal city known for gardens, mild weather, historic architecture, and waterfront scenery.",
        "best_season_to_visit": "Spring",
        "attractions": "Butchart Gardens, Inner Harbour, Beacon Hill Park",
        "budget": "Medium",
        "user_ratings": 4.6,
        "last_updated": "2026-10-04T00:00:00Z"
    },
    {
        "place": "Niagara Falls",
        "state": "Ontario",
        "description": "A popular destination centered around massive waterfalls, scenic viewpoints, entertainment, and nearby nature.",
        "best_season_to_visit": "Spring",
        "attractions": "Horseshoe Falls, Niagara Parkway, Journey Behind the Falls",
        "budget": "Low",
        "user_ratings": 4.5,
        "last_updated": "2026-10-04T00:00:00Z"
    },

    # ----- SUMMER -----

    {
        "place": "Halifax",
        "state": "Nova Scotia",
        "description": "A coastal city with maritime history, seafood, waterfront walks, and easy access to Atlantic scenery.",
        "best_season_to_visit": "Summer",
        "attractions": "Halifax Waterfront, Citadel Hill, Peggy's Cove",
        "budget": "Medium",
        "user_ratings": 4.5,
        "last_updated": "2026-10-04T00:00:00Z"
    },
    {
        "place": "Prince Edward Island",
        "state": "Prince Edward Island",
        "description": "A relaxed island destination with beaches, coastal drives, seafood, and inexpensive outdoor activities.",
        "best_season_to_visit": "Summer",
        "attractions": "Cavendish Beach, Green Gables, Confederation Trail",
        "budget": "Low",
        "user_ratings": 4.7,
        "last_updated": "2026-10-04T00:00:00Z"
    },

    # ----- FALL -----

    {
        "place": "Algonquin Provincial Park",
        "state": "Ontario",
        "description": "A large wilderness park famous for colorful autumn forests, hiking, canoeing, and wildlife.",
        "best_season_to_visit": "Fall",
        "attractions": "Lookout Trail, Canoe Lake, Highway 60 Corridor",
        "budget": "Low",
        "user_ratings": 4.8,
        "last_updated": "2026-10-04T00:00:00Z"
    },
    {
        "place": "Cape Breton Island",
        "state": "Nova Scotia",
        "description": "A scenic island known for spectacular autumn colors, coastal mountains, hiking, and road trips.",
        "best_season_to_visit": "Fall",
        "attractions": "Cabot Trail, Cape Breton Highlands, Skyline Trail",
        "budget": "Medium",
        "user_ratings": 4.7,
        "last_updated": "2026-10-04T00:00:00Z"
    }
]

joblib.dump(data, "data.joblib")

print("Saved", len(data), "places to data.joblib")