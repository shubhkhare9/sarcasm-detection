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
# NOTE: Keep these clearly absurd/satirical - avoid phrasing too close to real news
BABYLONBEE_HEADLINES = [
    # Clearly absurd - safe to train on
    "kamala harris 'thinking about' losing again in 2028",
    "biden announces he will continue serving as president in event of his death",
    "cnn purchases industrial-sized washing machine to spin news before publication",
    "trump awards presidential medal of freedom to trump",
    "aoc unveils new green new deal that will ban everything fun",
    "babylon bee writers forced to take week off as reality too absurd to parody",
    "man who has read entire bible somehow still worst person in office",
    "nation's last sane person requests transfer to different nation",
    "local man certain he could fix economy if given 15 minutes and a whiteboard",
    "area congressman introduces bill he definitely read",
    "new study finds 100% of studies sponsored by companies that benefit from study results",
    "man who just discovered politics very concerned you haven't heard his takes yet",
    "senator assures constituents he will get to their concerns right after his 14th fundraiser this week",
    "nation's pundits declare this the most important election of our lifetime for 8th consecutive election",
    "man confidently misquotes founding fathers to win facebook argument",
    "cnn breaking news alert: something happened somewhere involving someone",
    "politician who caused problem now heroically offering to fix problem",
    "area man's political views happen to align perfectly with his own self-interest",
    "new poll finds americans want politicians to work together except on anything they personally disagree with",
    "government announces new program to study why previous government programs didn't work",
    "local man's 3am tweet solves problem experts have struggled with for decades",
    "congress takes bold action by forming committee to discuss forming another committee",
    "man who hasn't read bill very passionate about bill",
    "nation somehow surprised that guy who lied about everything else also lied about this",
    "media discovers nuance, immediately loses it again",
    "politician spotted reading constitution for first time",
    "area voter shocked to learn candidate was politician all along",
    "new bipartisan bill unites both parties in agreeing it won't pass",
    "man who gets all news from memes very confident in his geopolitical analysis",
    "government efficiency task force adds 200 new employees to study inefficiency",
    "senator explains he can't comment on corruption investigation due to ongoing corruption",
    "nation's op-ed writers declare current moment unprecedented for 400th consecutive week",
    "local man's hot take on complex issue ready just 4 minutes after hearing about it",
    "politician promises to drain swamp immediately after attending swamp fundraiser",
    "breaking: man who was wrong about everything last year back with new predictions",
    "area think tank releases report confirming what think tank already believed",
    "congressman introduces legislation he will immediately forget about",
    "nation's cable news hosts agree: the other cable news hosts are the real problem",
    "man who has never run a business explains why all businesses are doing it wrong",
    "new government report finds government needs more funding to study why it needs more funding",
    "local pastor assures congregation god is on their side specifically",
    "area man's constitutional rights end exactly where his inconvenience begins",
    "politician who voted for war very moved by memorial day ceremony",
    "nation's most confident people also nation's least informed people, study finds",
    "man who just learned word 'gaslighting' now sees it everywhere",
    "breaking: expert who predicted last 0 of 10 recessions back with new recession prediction",
    "area woman's social media post about kindness gets 200 mean comments",
    "politician's memoir reveals he was the hero of every story he was in",
    "new app lets users feel outraged about things they don't fully understand faster than ever",
    "man who demands government stay out of his life also demands government fix everything",
    "nation's most powerful people very concerned about threat posed by nation's least powerful people",
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
