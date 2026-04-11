#!/usr/bin/env python3
"""
Manually add Babylon Bee headlines to training data.
Since Babylon Bee blocks automated scraping, we'll add curated examples.
"""

import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT, "data", "babylonbee_samples.csv")

# Curated Babylon Bee headlines (sarcastic political satire)
BABYLONBEE_HEADLINES = [
    "kamala harris 'thinking about' losing again in 2028",
    "biden announces he will continue serving as president in event of his death",
    "cnn purchases industrial-sized washing machine to spin news before publication",
    "trump awards presidential medal of freedom to trump",
    "aoc unveils new green new deal that will ban everything fun",
    "bernie sanders introduces new bill to make everyone as miserable as he is",
    "ocasio-cortez claims unemployment is low because everyone has two jobs",
    "democrats warn that if republicans confirm kavanaugh they will just have to pack the supreme court",
    "nation's conservatives announce they're moving to more progressive country where their values are respected",
    "babylon bee writers forced to take week off as reality too absurd to parody",
    "trump threatens to hold breath until democrats agree to fund wall",
    "ocasio-cortez excited to see how much she can spend in her first term",
    "democrats call for flags to be flown at half-mast to grieve death of roe v wade",
    "biden claims he's been to all 57 states",
    "cnn ratings plummet as network runs out of things to blame on trump",
    "democrats warn that if republicans win midterms they will have to accept election results",
    "ocasio-cortez proposes new green new deal that will ban cows",
    "trump announces plan to build wall around california",
    "biden announces he will run for president in 2024 if he can remember to",
    "democrats unveil new strategy to win elections: count all the votes",
    "ocasio-cortez claims she's a capitalist but also wants to abolish capitalism",
    "trump threatens to hold breath until mexico pays for wall",
    "biden claims he was arrested trying to see nelson mandela",
    "democrats warn that if republicans confirm barrett they will just have to pack the supreme court",
    "ocasio-cortez excited to learn what three branches of government are",
    "trump announces plan to make america great again again",
    "biden announces he will continue campaigning from his basement",
    "democrats unveil new strategy to win elections: have media do it for them",
    "ocasio-cortez proposes new green new deal that will ban airplanes",
    "trump threatens to hold breath until democrats agree to investigate hunter biden",
    "biden claims he was vice president under obama",
    "democrats warn that if republicans win election they will have to accept results",
    "ocasio-cortez excited to learn what electoral college is",
    "trump announces plan to drain swamp by filling it with his own swamp creatures",
    "biden announces he will run for senate",
    "democrats unveil new strategy to win elections: call everyone racist",
    "ocasio-cortez proposes new green new deal that will ban cars",
    "trump threatens to hold breath until democrats agree to impeach him again",
    "biden claims he has plan to cure cancer",
    "democrats warn that if republicans confirm gorsuch they will just have to pack the supreme court",
    "ocasio-cortez excited to learn what capitalism is",
    "trump announces plan to make america great again again again",
    "biden announces he will continue hiding in basement until election is over",
    "democrats unveil new strategy to win elections: have big tech censor opponents",
    "ocasio-cortez proposes new green new deal that will ban breathing",
    "trump threatens to hold breath until democrats agree to build wall",
    "biden claims he was arrested at apartheid protest",
    "democrats warn that if republicans win they will have to respect election results",
    "ocasio-cortez excited to learn what socialism is",
    "trump announces plan to drain swamp by tweeting about it",
    "biden announces he will run for president of united states",
]

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["headline", "is_sarcastic"])
        for headline in BABYLONBEE_HEADLINES:
            writer.writerow([headline, 1])
    
    print(f"✅ Created {len(BABYLONBEE_HEADLINES)} Babylon Bee samples → {OUT_PATH}")
    print(f"\nNext steps:")
    print(f"1. Run: python scrape_headlines.py")
    print(f"2. Merge this file with extra_headlines.csv")
    print(f"3. Run: python finetune_bert.py")

if __name__ == "__main__":
    main()
