import re
import time
import requests

TITLES = [
    "History of India", "World War II", "French Revolution", "Roman Empire",
    "Industrial Revolution", "Climate change", "Evolution", "Solar System",
    "Quantum mechanics", "Human evolution", "Ancient Egypt", "Renaissance",
    "Cold War", "Great Depression", "Mughal Empire", "Ocean",
    "Photosynthesis", "Immune system", "Electricity", "Internet",
    "History of China", "Chemistry", "Plate tectonics", "Mahatma Gandhi",
    "Amazon rainforest",
]

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "context-stress-tester/0.1 (student project)"}

for title in TITLES:
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": title,
    }
    try:
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        page = next(iter(r.json()["query"]["pages"].values()))
        text = page.get("extract", "")
    except Exception as e:
        print(f"FAILED {title}: {e}")
        continue
    if len(text) < 5000:
        print(f"SKIPPED (too short or missing): {title}")
        continue
    filename = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    with open(f"data/sources/wiki_{filename}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {title} ({len(text)} characters)")
    time.sleep(1)