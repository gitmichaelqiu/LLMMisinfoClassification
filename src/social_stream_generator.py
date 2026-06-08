"""Synthetic social media stream generator for MFT temporal events.

Generates 10-20 timestamped tweets per event_id between T0 (0s) and
T2 (300s) that model realistic social media dynamics:

- FAKE events (label=1): rising 'debunk' velocity — initial panic,
  then skepticism, then confirmed-hoax tweets as T2 approaches.
- REAL events (label=0): rising 'amplification' velocity — initial
  breaking news, then corroboration, then official confirmation.

Output: output/social_stream.csv
"""

import os
import csv
import random
import numpy as np
import pandas as pd

# --- Synthetic user base ---
AUTHOR_POOL = [
    "@MarketMaven", "@TradeAlertBot", "@FinanceGuru", "@CryptoWhale",
    "@WallStWatcher", "@MacroMinded", "@EquityEdge", "@DerivativesDude",
    "@QuantQueen", "@RiskMetrics", "@FlowTracker", "@NewsCatalyst",
    "@EarningsSleuth", "@ShortSellerPro", "@LongOnlyLP", "@VolArbVic",
    "@TheRealSkeptic", "@FactCheckFinance", "@DeepDiveDave", "@OCC_Alert",
    "@SatelliteView", "@ArbClub", "@HFT_Hunter", "@LiquidityLens",
    "@FakeNewsFilter", "@VerifiedSource", "@MarketSentinel",
]

# --- Phased tweet template pools ---

PANIC_TEMPLATES = [
    "🚨 {headline_short} SELL EVERYTHING",
    "WTF {entity} just dropped ${drop_pct} on this news",
    "This {entity} news is insane, book down ${drop_amt}k already",
    "SOMEONE TELL ME THIS ISN'T TRUE {headline_short}",
    "My portfolio is melting down over {headline_short}",
    "Can't believe {entity} is getting crushed right now",
    "Watching {ticker} plummet on this news, stop loss triggered",
    "Who saw this {entity} news coming? Market in full panic mode",
    "Liquidity drying up on {ticker}, spread is enormous",
    "This is going to be a bloodbath for {entity} holders",
]

SKEPTICISM_TEMPLATES = [
    "Hang on, has anyone verified the {entity} news source?",
    "Something feels off about this {entity} headline...",
    "Wait, this {entity} story — who originally broke it?",
    "Can't find any corroboration for {headline_short}",
    "Looking at the tape, this {entity} move seems like algos overreacting",
    "Anyone else getting suspicious about {headline_short}?",
    "The speed of this {entity} move feels wrong, not organic",
    "I'm seeing conflicting reports about {headline_short}",
    "Has {entity} put out a statement yet on this?",
    "This {entity} headline contradicts everything I know about them",
]

DEBUNK_TEMPLATES = [
    "CONFIRMED: {headline_short} is FALSE. Source was spoofed.",
    "UPDATE: {entity} official statement denies {headline_short} as fabricated",
    "Just checked — {headline_short} is a doctored headline from {fake_source}",
    "Fact check: {entity} numbers in that article are completely made up",
    "This {entity} news is confirmed FAKE — original source just retracted",
    "We've seen this hoax pattern before — {headline_short} is fabricated",
    "Official {entity} PR: '{denial_text}'",
    "Whoever manufactured this {entity} story used a fake timestamp",
    "The filing referenced in {headline_short} doesn't exist on EDGAR",
    "This was a deliberate misinformation attack on {entity} stock",
]

BREAKING_TEMPLATES = [
    "🚨 BREAKING: {headline_short}",
    "JUST IN: {entity} news hitting the tape",
    "Alert: {headline_short} — market reacting",
    "First report: {headline_short}",
    "Developing story on {entity}: {headline_short}",
    "Breaking: {entity} stock moving on {headline_short}",
    "News flash: {headline_short} — details to follow",
    "Big {entity} news coming across the wire: {headline_short}",
    "Market alert — {entity} triggered by {headline_short}",
    "Headline just crossed: {headline_short}",
]

AMPLIFICATION_TEMPLATES = [
    "Confirmed by multiple sources: {headline_short}",
    "{entity} story is being picked up by all major outlets",
    "More details emerging on {entity}: {amplification_detail}",
    "This lines up with what we've been hearing about {entity}",
    "Other sources now confirming the {entity} report",
    "Context on {entity} news: {amplification_detail}",
    "This is consistent with the {entity} thesis I've been tracking",
    "Add this to the mounting evidence on {entity}",
    "Additional reporting confirms {headline_short}",
    "Market digesting {entity} news — volume confirms genuine reaction",
]

CONFIRMATION_TEMPLATES = [
    "OFFICIAL: {entity} confirms {headline_short} via press release",
    "Statement from {entity}: '{official_statement_text}'",
    "Verified: {headline_short} is accurate — confirmed by {entity} IR",
    "{entity} filing now on EDGAR confirms the news",
    "{regulatory_source} filing confirms {headline_short}",
    "Later reporting validates the initial {entity} story",
    "UPDATE: The {entity} news is real — confirmed by independent verification",
    "Official numbers are in for {entity} — consistent with initial report",
    "This {entity} event is genuine — multiple independent confirmations",
    "Confirmed real — {entity} story checked out by {analyst_source} analysis",
]

# Short headline extract
def _shorten(headline, max_chars=60):
    if len(headline) <= max_chars:
        return headline
    return headline[:max_chars-3].rsplit(" ", 1)[0] + "..."

# Fake source names for hoax attribution
FAKE_SOURCES = ["a spoofed SEC filing", "a doctored Bloomberg terminal screenshot",
                "a fake press release", "a compromised news wire",
                "an old article with altered date", "a parody account"]

# Amplification details
AMPLIFICATION_DETAILS = [
    "the leak source has been identified within the company",
    "employee chatter confirms internal discussions",
    "supplier orders show a material increase",
    "historical data shows this pattern preceded similar events",
    "options flow confirms institutional positioning",
    "management was seen meeting with advisors last week",
    "regulatory filings show the application was submitted",
    "the Board has a scheduled meeting to discuss this",
]

# Official statements for real events
OFFICIAL_STATEMENTS = [
    "We confirm the information released earlier today",
    "We are pleased to share this update with our stakeholders",
    "This reflects our strategic direction as previously communicated",
    "We can confirm this is consistent with our public guidance",
    "This development is the result of our ongoing initiatives",
]

# Denial texts for fake events
DENIAL_TEXTS = [
    "This report is completely false and we are exploring legal options",
    "We have not issued any such statement; this is a hoax",
    "The information circulating is fabricated and does not reflect our operations",
    "We categorically deny the claims in this fabricated news report",
    "This is a doctored version of an old press release from 2023",
]

# ── Adversarial Bot Templates ──────────────────────────────────

BOT_AGREE_TEMPLATES = [
    "I have inside sources confirming {headline_short} is 100% legit",
    "My contacts at {entity} verified this an hour ago, it's real",
    "People saying this is fake are going to look foolish when the official statement drops",
    "The debunkers are wrong — I've seen the internal memo myself",
    "{entity} employee on a private channel just confirmed this is accurate",
    "This isn't fake, the shorts are spreading FUD to cover their positions",
    "I've been following {entity} for years, this timing is consistent with their roadmap",
    "The panic sellers will regret this when the news is officially confirmed",
]

BOT_ATTACK_TEMPLATES = [
    "The accounts calling this fake are known bots spreading misinformation",
    "Seeing coordinated debunk attempts means the news is REAL and someone is scared",
    "Look at the history of these 'debunk' accounts — they're all shills",
    "Don't listen to the FUDsters, they do this every time {entity} has good news",
    "The fake news accusations are from people who can't handle the truth about {entity}",
    "This coordinated debunk is obviously from {entity}'s competitors",
    "Anyone claiming this is fake hasn't done their own research on {entity}",
    "The 'skeptics' in these replies have zero credibility",
]

BOT_AUTHORITY_TEMPLATES = [
    "Just checked with my source at {regulatory_source}, the {entity} filing is legitimate",
    "I have a friend on the {entity} board who confirms this is accurate",
    "The official statement will be out within hours confirming everything",
    "Legal review has already cleared the {entity} announcement — it's real",
    "My firm's compliance department just confirmed the {entity} news is authentic",
    "The data room shows this {entity} deal has been in progress for months",
    "Every major outlet is picking this up now — it's confirmed",
    "The filing reference number is legitimate, I checked the {regulatory_source} system",
]

BOT_DIVERSION_TEMPLATES = [
    "Why is everyone focused on {entity} when the real story is the broader market manipulation?",
    "This is a coordinated attack by short sellers, {entity} is a solid company",
    "The real fraud here is the media suppressing positive {entity} news",
    "Mark my words, {entity} will release a statement that makes the skeptics look ridiculous",
    "This exact playbook was used against {other_company} last year — fake debunk to shake out retail",
]


def generate_social_stream(temporal_events_path="./output/temporal_events.csv",
                           output_path="./output/social_stream.csv",
                           seed=42):
    """Generate synthetic social media posts for each temporal event.

    For each event in temporal_events.csv, produces 10-20 timestamped
    tweets between T0 (0s) and T2 (300s) that follow realistic dynamics:
    - FAKE events: panic → skepticism → debunk (rising debunk velocity)
    - REAL events: breaking → amplification → confirmation (rising amplification)

    Args:
        temporal_events_path: Path to temporal_events.csv
        output_path: Output CSV path
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame with columns: post_id, event_id, timestamp_seconds,
        author, text, post_type
    """
    events_df = pd.read_csv(temporal_events_path)
    rng = np.random.default_rng(seed)
    py_random = random.Random(seed)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    posts = []
    post_counter = 0

    for _, event in events_df.iterrows():
        event_id = event["event_id"]
        headline = str(event["T0_headline"])
        label = int(event["T2_human_verdict"])
        entity = str(event.get("entity", "UNKNOWN"))
        sector = str(event.get("sector", "Macro"))

        short = _shorten(headline)
        ticker = entity[:4].upper() if entity != "UNKNOWN" else "MKTS"
        headline_short = short.lower()
        fake = (label == 1)

        # Number of tweets per event: 12-20
        n_posts = py_random.randint(12, 20)

        # Assign tweets to time phases
        timestamps = sorted(rng.uniform(0, 300, n_posts))

        for t in timestamps:
            post_counter += 1
            author = py_random.choice(AUTHOR_POOL)

            if fake:
                # FAKE event dynamics: panic → skepticism → debunk
                if t < 30:
                    post_type = "panic"
                    template = py_random.choice(PANIC_TEMPLATES)
                    drop_pct = py_random.randint(5, 25)
                    drop_amt = py_random.randint(5, 50)
                    text = template.format(
                        headline_short=headline_short,
                        entity=entity,
                        ticker=ticker,
                        drop_pct=drop_pct,
                        drop_amt=drop_amt,
                    )
                elif t < 120:
                    post_type = "skepticism"
                    template = py_random.choice(SKEPTICISM_TEMPLATES)
                    text = template.format(
                        headline_short=headline_short,
                        entity=entity,
                        ticker=ticker,
                    )
                else:
                    post_type = "debunk"
                    template = py_random.choice(DEBUNK_TEMPLATES)
                    fake_source = py_random.choice(FAKE_SOURCES)
                    denial_text = py_random.choice(DENIAL_TEXTS)
                    text = template.format(
                        headline_short=headline_short,
                        entity=entity,
                        fake_source=fake_source,
                        denial_text=denial_text,
                    )
            else:
                # REAL event dynamics: breaking → amplification → confirmation
                if t < 30:
                    post_type = "breaking"
                    template = py_random.choice(BREAKING_TEMPLATES)
                    text = template.format(
                        headline_short=headline_short,
                        entity=entity,
                        ticker=ticker,
                    )
                elif t < 160:
                    post_type = "amplification"
                    template = py_random.choice(AMPLIFICATION_TEMPLATES)
                    amp_detail = py_random.choice(AMPLIFICATION_DETAILS)
                    text = template.format(
                        headline_short=headline_short,
                        entity=entity,
                        ticker=ticker,
                        amplification_detail=amp_detail,
                    )
                else:
                    post_type = "confirmation"
                    template = py_random.choice(CONFIRMATION_TEMPLATES)
                    official_text = py_random.choice(OFFICIAL_STATEMENTS)
                    regulatory_source = py_random.choice(["SEC", "FCA", "CFTC", "FINRA", "EDGAR"])
                    analyst_source = py_random.choice(["Bloomberg Intelligence", "Goldman Sachs", "JPMorgan", "independent"])
                    text = template.format(
                        headline_short=headline_short,
                        entity=entity,
                        ticker=ticker,
                        official_statement_text=official_text,
                        regulatory_source=regulatory_source,
                        analyst_source=analyst_source,
                    )

            posts.append({
                "post_id": f"POST-{post_counter:06d}",
                "event_id": event_id,
                "timestamp_seconds": round(t, 2),
                "author": author,
                "text": text,
                "post_type": post_type,
                "is_fake_event": int(fake),
                "entity": entity,
                "sector": sector,
            })

    posts_df = pd.DataFrame(posts)
    posts_df.to_csv(output_path, index=False)
    print(f"Generated {len(posts_df)} social media posts to {output_path}")
    print(f"  Events covered: {posts_df['event_id'].nunique()}")
    print(f"  Post types: {posts_df['post_type'].value_counts().to_dict()}")
    print(f"  Avg posts per event: {len(posts_df) / posts_df['event_id'].nunique():.1f}")
    return posts_df


def compute_debunk_velocity(social_stream_df, event_id):
    """Compute the debunk-velocity curve for a single event.

    Returns a dict with time-binned counts of debunk vs non-debunk posts
    for analyzing the temporal pattern.
    """
    event_posts = social_stream_df[social_stream_df["event_id"] == event_id].copy()
    if event_posts.empty:
        return {}

    bins = [0, 30, 60, 120, 180, 240, 300]
    labels = ["0-30s", "30-60s", "60-120s", "120-180s", "180-240s", "240-300s"]
    event_posts["time_bin"] = pd.cut(event_posts["timestamp_seconds"], bins=bins, labels=labels, right=False)

    velocity = {}
    for label in labels:
        bin_posts = event_posts[event_posts["time_bin"] == label]
        n_total = len(bin_posts)
        n_debunk = len(bin_posts[bin_posts["post_type"] == "debunk"])
        n_confirm = len(bin_posts[bin_posts["post_type"] == "confirmation"])
        velocity[label] = {
            "total_posts": n_total,
            "debunk_posts": n_debunk,
            "confirmation_posts": n_confirm,
            "debunk_ratio": round(n_debunk / n_total, 3) if n_total > 0 else 0,
        }
    return velocity


# ── Adversarial Bot Accounts ────────────────────────────────────

BOT_AUTHORS = [
    "@InsiderScoop_REAL", "@VerifiedSource_Bot", "@DeepState_Alpha",
    "@TheTruthSpreader", "@Whistleblower_Off", "@ConfidentialInfo",
    "@LeakedDocuments", "@ClassifiedDeal", "@BoardRoom_Leaks",
    "@ExecutiveBrief", "@InsiderTrack_net", "@CorporateIntelHub",
    "@UndergroundSource", "@DarkPool_Alpha", "@WhaleAlley_Insider",
    "@MarketMaker_Prop", "@InstitutionalFlow_", "@SovereignFund_View",
    "@HedgeFund_Signal", "@MacroAlpha_Watch",
]


def generate_adversarial_stream(
    temporal_events_path="./output/temporal_events.csv",
    output_path=None,
    seed=42,
    bot_pct=0.5,
):
    """Generate a poisoned social media stream with adversarial bot activity.

    Injects bot accounts that aggressively defend fake news, creating
    conflicting consensus signals to stress-test the LLM verifier.

    Normal FAKE stream: panic → skepticism → debunk
    Adversarial FAKE stream: panic + bots_defending → skepticism + bots_attacking
                                                    → debunk + bots_diversion

    Args:
        temporal_events_path: Path to temporal_events.csv
        output_path: Output CSV path (default: social_stream_adversarial_{bot_pct}.csv)
        seed: Random seed
        bot_pct: Fraction of posts generated by bots (0.0 to 1.0)

    Returns:
        pd.DataFrame with adversarial social stream
    """
    import random as _random

    if output_path is None:
        bot_label = str(int(bot_pct * 100))
        output_path = f"./output/social_stream_adversarial_bots{bot_label}.csv"

    base_df = generate_social_stream(
        temporal_events_path=temporal_events_path,
        output_path=output_path,
        seed=seed,
    )

    # No adversarial modification needed at 0% intensity
    if bot_pct <= 0.0:
        return base_df

    rng = np.random.default_rng(seed + 999)
    py_random = _random.Random(seed + 999)

    events_df = pd.read_csv(temporal_events_path)

    adversarial_posts = []
    post_counter = len(base_df) + 1

    for _, event in events_df.iterrows():
        event_id = event["event_id"]
        label = int(event["T2_human_verdict"])
        entity = str(event.get("entity", "UNKNOWN"))
        headline = str(event["T0_headline"])
        sector = str(event.get("sector", "Macro"))

        # Only poison fake events (adversarial attack targets fake news)
        if label != 1:
            continue

        short = _shorten(headline)
        ticker = entity[:4].upper() if entity != "UNKNOWN" else "MKTS"
        headline_short = short.lower()

        # Determine how many bot posts to add
        base_posts = base_df[base_df["event_id"] == event_id]
        n_base = len(base_posts)
        n_bots = max(1, int(n_base * bot_pct / (1 - bot_pct + 1e-9)))

        for _ in range(n_bots):
            post_counter += 1
            author = py_random.choice(BOT_AUTHORS)
            t = rng.uniform(30, 280)  # bots active during skepticism/debunk phase

            # Select bot post type
            phase_roll = rng.random()
            regulatory_source = py_random.choice(["SEC", "FCA", "CFTC", "FINRA", "EDGAR"])
            other_company = py_random.choice(["Enron", "FTX", "Wirecard", "Luckin Coffee", "Nikola"])

            if t < 100:
                # Early phase: aggressive agreement/amplification of fake news
                if phase_roll < 0.6:
                    template = py_random.choice(BOT_AGREE_TEMPLATES)
                    text = template.format(
                        headline_short=headline_short, entity=entity,
                        regulatory_source=regulatory_source,
                    )
                else:
                    template = py_random.choice(BOT_AUTHORITY_TEMPLATES)
                    text = template.format(
                        entity=entity, regulatory_source=regulatory_source,
                    )
                post_type = "bot_amplify"
            elif t < 200:
                # Mid phase: attack debunkers
                if phase_roll < 0.5:
                    template = py_random.choice(BOT_ATTACK_TEMPLATES)
                    text = template.format(entity=entity)
                elif phase_roll < 0.8:
                    template = py_random.choice(BOT_AUTHORITY_TEMPLATES)
                    text = template.format(
                        entity=entity, regulatory_source=regulatory_source,
                    )
                else:
                    template = py_random.choice(BOT_DIVERSION_TEMPLATES)
                    text = template.format(
                        entity=entity, other_company=other_company,
                    )
                post_type = "bot_attack"
            else:
                # Late phase: diversion and authority claims
                if phase_roll < 0.5:
                    template = py_random.choice(BOT_DIVERSION_TEMPLATES)
                    text = template.format(
                        entity=entity, other_company=other_company,
                    )
                else:
                    template = py_random.choice(BOT_AUTHORITY_TEMPLATES)
                    text = template.format(
                        entity=entity, regulatory_source=regulatory_source,
                    )
                post_type = "bot_diversion"

            adversarial_posts.append({
                "post_id": f"POST-A{post_counter:06d}",
                "event_id": event_id,
                "timestamp_seconds": round(t, 2),
                "author": author,
                "text": text,
                "post_type": post_type,
                "is_fake_event": 1,
                "entity": entity,
                "sector": sector,
                "is_bot": 1,
            })

    adv_df = pd.DataFrame(adversarial_posts)

    if len(adv_df) > 0:
        # Merge adversarial posts with base stream
        merged = pd.concat([base_df, adv_df], ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"\n[Adversarial] Generated {len(adv_df)} adversarial bot posts ({bot_pct:.0%} intensity)")
        print(f"[Adversarial] Total stream: {len(merged)} posts → {output_path}")
    else:
        print("[Adversarial] No adversarial posts generated (no fake events in dataset)")

    return merged if len(adv_df) > 0 else base_df


if __name__ == "__main__":
    df = generate_social_stream()
    for eid in df["event_id"].unique()[:3]:
        v = compute_debunk_velocity(df, eid)
        print(f"\nDebunk velocity for {eid}:")
        for tb, stats in v.items():
            print(f"  {tb}: {stats}")

    print("\n" + "=" * 60)
    print("Generating adversarial streams at various intensities...")
    print("=" * 60)
    for bot_level in [0.25, 0.50, 0.75]:
        generate_adversarial_stream(bot_pct=bot_level)
