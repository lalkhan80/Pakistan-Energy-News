"""Approved newspaper sources and energy vocabulary for the Pakistan Energy News Agent."""

SOURCES = [
    {
        "name": "Dawn",
        "base_url": "https://www.dawn.com/",
        "listing_urls": [
            "https://www.dawn.com/latest-news",
            "https://www.dawn.com/business",
        ],
    },
    {
        "name": "Business Recorder",
        "base_url": "https://www.brecorder.com/",
        "listing_urls": [
            "https://www.brecorder.com/business-finance",
            "https://www.brecorder.com/markets/energy",
            "https://www.brecorder.com/pakistan",
        ],
    },
    {
        "name": "The News International",
        "base_url": "https://www.thenews.com.pk/",
        "listing_urls": [
            "https://www.thenews.com.pk/latest/category/business",
            "https://www.thenews.com.pk/print/category/business",
        ],
    },
    {
        "name": "The Express Tribune",
        "base_url": "https://tribune.com.pk/",
        "listing_urls": [
            "https://tribune.com.pk/business",
            "https://tribune.com.pk/business/archives",
        ],
    },
]

CATEGORIES = [
    "Petroleum",
    "OMC",
    "Refinery",
    "Gas / LNG / LPG",
    "Power",
    "Energy Policy",
]

# Broad first-pass terms. The AI agent performs the final relevance decision.
ENERGY_TERMS = [
    "oil", "crude", "petrol", "gasoline", "diesel", "hsd", "pmg", "fuel",
    "petroleum", "pol", "petroleum levy", "furnace oil", "jet fuel",
    "omc", "oil marketing", "pso", "pakistan state oil", "attock petroleum",
    "wafi", "shell pakistan", "go petroleum",
    "refinery", "refineries", "parco", "attock refinery", "arl",
    "pakistan refinery", "prl", "national refinery", "nrl", "cnergyico",
    "natural gas", "lng", "rlng", "lpg", "sngpl", "ssgc",
    "ogra", "petroleum division", "ministry of energy",
    "power", "electricity", "tariff", "nepra", "cppa", "ntdc", "disco",
    "ipp", "circular debt", "load-shedding", "load shedding",
    "solar", "hydel", "hydropower", "wind energy", "renewable energy",
    "k-electric", "k electric", "power generation", "transmission",
    "energy sector", "energy policy", "energy crisis", "energy supply",
    "ogdcl", "oil and gas development", "ppl", "pakistan petroleum",
    "mari energies", "exploration", "gas discovery",
]

EXCLUDED_PATH_PARTS = [
    "/authors/", "/author/", "/tags/", "/tag/", "/topics/", "/topic/",
    "/images/", "/photos/", "/videos/", "/video/", "/rss", "/epaper",
    "/contact", "/about", "/privacy", "/sports/", "/entertainment/",
]
