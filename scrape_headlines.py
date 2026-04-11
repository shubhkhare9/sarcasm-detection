#!/usr/bin/env python3
"""
Scrape extra headlines for fine-tuning:
  Sarcastic (label=1): The Beaverton, NewsThump, Waterford Whispers,
                       The Shovel, Reductress, ClickHole
  Real news (label=0): NPR, BBC, NYT, The Guardian

Output: data/extra_headlines.csv  [headline, is_sarcastic]
"""

import ssl
import certifi
import time
import random
import csv
import os
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request

ROOT     = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT, "data", "extra_headlines.csv")

CTX = ssl.create_default_context(cafile=certifi.where())

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1)"}

SATIRE_FEEDS = [
    "https://www.thebeaverton.com/feed/",
    "https://newsthump.com/feed/",
    "https://waterfordwhispersnews.com/feed/",
    "https://www.theshovel.com.au/feed/",
    "https://reductress.com/feed/",
    "https://clickhole.com/feed/",
]

REAL_FEEDS = [
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.npr.org/1004/rss.xml",
    "https://feeds.npr.org/1007/rss.xml",
    "https://feeds.npr.org/1008/rss.xml",
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/us-news/rss",
    "https://www.theguardian.com/technology/rss",
    "https://www.theguardian.com/science/rss",
    "https://www.theguardian.com/business/rss",
]

# Paginated feeds — append ?paged=N
PAGINATED_SATIRE = [
    "https://www.thebeaverton.com/feed/",
    "https://newsthump.com/feed/",
    "https://waterfordwhispersnews.com/feed/",
]


def fetch(url):
    req = Request(url, headers=HEADERS)
    with urlopen(req, context=CTX, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


def clean(text):
    return " ".join(text.lower().strip().split())


def parse_rss_titles(xml_text, skip_keywords=None):
    skip_keywords = skip_keywords or []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    titles = []
    for item in root.findall(".//item"):
        el = item.find("title")
        if el is not None and el.text:
            t = el.text.strip()
            if len(t) > 15 and not any(k in t.lower() for k in skip_keywords):
                titles.append(clean(t))
    return titles


def scrape_feeds(feeds, label_name, skip_keywords=None, pages=1):
    headlines = []
    for url in feeds:
        for page in range(1, pages + 1):
            paged_url = f"{url}?paged={page}" if page > 1 else url
            try:
                xml_text = fetch(paged_url)
                batch = parse_rss_titles(xml_text, skip_keywords)
                headlines.extend(batch)
                print(f"  [{label_name}] {paged_url.split('/')[2]} p{page}: {len(batch)} headlines")
                time.sleep(random.uniform(0.6, 1.2))
            except Exception as e:
                print(f"  [{label_name}] FAIL {paged_url}: {e}")
                break   # stop paginating this feed on error
    return list(dict.fromkeys(headlines))  # deduplicate


def main():
    print("=== Scraping satire sources ===")
    # Single-page feeds
    single_satire = [f for f in SATIRE_FEEDS if f not in PAGINATED_SATIRE]
    sarcastic = scrape_feeds(single_satire, "satire", pages=1)
    # Paginated feeds — go deeper
    sarcastic += scrape_feeds(PAGINATED_SATIRE, "satire", pages=12)
    sarcastic = list(dict.fromkeys(sarcastic))

    print(f"\n=== Scraping real news sources ===")
    real = scrape_feeds(
        REAL_FEEDS,
        "real",
        skip_keywords=["subscribe", "newsletter", "podcast", "video"],
        pages=1,
    )

    n = min(len(sarcastic), len(real))
    if n == 0:
        print("\n❌ No headlines scraped. Check your internet connection.")
        return

    sarcastic = sarcastic[:n]
    real      = real[:n]

    rows = [(h, 1) for h in sarcastic] + [(h, 0) for h in real]
    random.shuffle(rows)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["headline", "is_sarcastic"])
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} headlines → {OUT_PATH}")
    print(f"   Sarcastic : {len(sarcastic)}")
    print(f"   Real news : {len(real)}")


if __name__ == "__main__":
    main()
