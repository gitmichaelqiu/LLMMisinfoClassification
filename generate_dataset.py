import pandas as pd
import random

# Categories for diversity
SECTORS = ["Tech", "Macro", "Banking", "Energy", "Retail", "Automotive", "Pharma"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY", "CHF"]
COUNTRIES = ["US", "UK", "Germany", "Japan", "China", "France", "Switzerland", "Canada", "India"]
ENTITIES = ["Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla", "NVIDIA", "JPMorgan", "Wells Fargo", "Goldman Sachs", 
            "ExxonMobil", "Chevron", "Pfizer", "Moderna", "Disney", "Ford", "Toyota", "Walmart", "Home Depot", "Nike",
            "FedEx", "Visa", "Mastercard", "Netflix", "Oracle", "IBM", "Intel", "Adobe", "Salesforce", "Verizon"]

AUTH_PATTERNS = [
    "{entity} reports {quarter} revenue of ${val}B, beating analyst estimates by {pct}%.",
    "Federal Reserve {action} benchmark interest rates at {range} to combat inflation.",
    "{country} annual Consumer Price Index (CPI) increases by {pct}% in {month}.",
    "{commodity} prices {direction} by {pct}% following {reason}.",
    "S&P 500 {direction} {pct}% as {sector} stocks lead the {trend}.",
    "{entity} to acquire {target} for ${billion}B in a strategic push into {subsector}.",
    "US Labor Department: Nonfarm payrolls increased by {jobs}k in {month}.",
    "{central_bank} signals potential rate {rate_move} in {month} after {inflation_type} inflation data.",
    "{bank} CEO warns of {economic_state} risk by late {year}.",
    "{entity} stock surges {pct}% after announcing new {product} for {market_segment}.",
    "OPEC+ agrees to {prod_move} oil output by {bpd}M barrels per day through {month}.",
    "Tech sell-off continues as {entity} shares drop {pct}% amid regulatory concerns.",
    "Bonds {direction} as yields hit {val}% following {event}.",
    "Retail sales in {country} {direction} {pct}% as consumer spending {consumer_state}.",
    "Goldman Sachs upgrades {entity} to 'Buy' with price target of ${val}.",
    "BlackRock AUM reaches ${billion}T as ETF inflows accelerate.",
    "{entity} announces {billion} share repurchase program starting {month}.",
    "Nikkei 225 {direction} {pct}% tracking {region} market gains.",
    "{currency} hits multi-month {direction} against {other_currency} after {policy_shift}.",
    "{entity} cuts {pct}% of workforce to improve margins in {year}."
]

ANOMALY_PATTERNS = [
    "{entity} files for Chapter 11 bankruptcy despite holding ${val}B in liquid cash reserves.",
    "Federal Reserve announces interest rates will be set at {impossible_val}% to 'halt economic time'.",
    "Microsoft announces acquisition of {government_entity} for ${valuation}T in cash.",
    "{commodity} prices reach ${extreme_val} per {unit} as demand drops to absolute zero globally.",
    "{country} GDP grew by {extreme_pct}% in the last {time_unit} following a {strange_event}.",
    "Warren Buffett liquidates his entire portfolio in {val} minutes to buy {obscure_asset}.",
    "Central Bank of {country} replaces its entire currency with {physical_object} overnight.",
    "Tesla reports it has successfully delivered 1 million vehicles to {outer_space_location}.",
    "NVIDIA CEO says AI has become {conscious_state}, making financial markets 'obsolete'.",
    "ECB moves all gold reserves to {secret_location} for 'mathematical safety'.",
    "Goldman Sachs replaces all human executives with {deprecated_tech} to 'reduce ego costs'.",
    "Oil prices hit ${val} per share as ExxonMobil converts all refineries into {strangepipe}.",
    "Amazon reports it now owns {extreme_pct}% of the Earth's total land area.",
    "Apple announces an iPhone that can 'mine gold from the surrounding air'.",
    "Federal Reserve to discontinue all paper money and transition to {unstable_currency}.",
    "S&P 500 reaches {extreme_val} points after the discovery of {impossible_resource}.",
    "JPMorgan announces it will no longer lend money, only 'hopes and dreams'.",
    "Bitcoin mining declared illegal by {government_entity} until it produces 'visible heat'.",
    "Meta announces it will rename the {planet} to 'The ZuckSphere' by {year}.",
    "Alphabet's CEO says Google will stop searching and 'begin knowing' everything by {month}."
]

REALISTIC_ANOMALY_PATTERNS = [
    "{entity} reports quarterly revenue of ${val}B, defying analyst consensus of ${val2}B despite {sector} downturn.",
    "{entity} announces ${billion}B buyback despite having only ${cash}B in cash reserves.",
    "Federal Reserve cuts rates to {low_val}% while core inflation remains at {high_val}%.",
    "{entity} acquires {target} for ${billion}B, exceeding {target} market cap of ${val}M.",
    "Unemployment in {country} drops to {low_val}%, historically low amid mass layoffs in {sector}.",
    "{country} GDP expands {pct}% in Q{q}, fastest quarterly growth on record during {economic_condition}.",
    "{entity} CEO sells ${val}M in stock days before {positive_event}, avoiding {pct}% loss.",
    "Oil plunges to ${val}/barrel as OPEC+ unexpectedly boosts output {bpd}M bpd.",
    "{currency} surges {pct}% versus {other_currency} in single session with no catalyst.",
    "{entity} market cap hits ${val}T, exceeding {competitor1} and {competitor2} combined.",
    "Consumer confidence rises to {val} while retail sales fall {pct}%, baffling economists.",
    "{bank} forecasts {country} recession by Q{q} yet raises equity allocation to {pct}%.",
    "ECB hikes rates to {val}% as Eurozone inflation cools to {low_val}%.",
    "{entity} net income jumps {pct}% despite revenue drop of {pct2}%.",
    "Gold hits ${val}/oz as central banks {action}, contradicting Q{q} demand reports.",
    "SEC charges {entity} with ${val}B fraud, surpassing {entity} annual revenue of ${val2}B.",
    "{company} plans {num} factories in {country}, which has zero {sector} workforce.",
    "{entity} operating margin surges {pct}% after {reason} while revenue stays flat.",
    "{country} sovereign debt yields turn negative across all tenors despite {pct}% inflation.",
    "Treasury yield curve inverts {val}bp, steepest since {year}, as economy {economic_state}.",
]

def generate_unique_df(count=200):
    random.seed(42)
    unique_headlines = set()
    rows = []

    def pick_choice(name, *args, **kwargs):
        return random.choice(*args, **kwargs)

    # 100 Authentic
    while len([r for r in rows if r['label'] == 0]) < 100:
        p = random.choice(AUTH_PATTERNS)
        h = p.format(
            entity=random.choice(ENTITIES),
            quarter=random.choice(["Q1", "Q2", "Q3", "Q4", "first-quarter", "third-quarter"]),
            val=random.randint(10, 250),
            pct=random.randint(2, 40),
            action=random.choice(["maintains", "hikes", "lowers", "holds"]),
            range=random.choice(["5.25%-5.50%", "3.00-3.50%", "0.00-0.25%"]),
            country=random.choice(COUNTRIES),
            month=random.choice(["January", "March", "June", "September", "November"]),
            commodity=random.choice(["Oil", "Natural Gas", "Gold", "Copper", "Wheat", "Coffee"]),
            direction=random.choice(["rises", "falls", "slips", "soared", "plunged"]),
            reason=random.choice(["supply chain issues", "geopolitical tensions", "OPEC decision", "US reserve news"]),
            sector=random.choice(SECTORS),
            trend=random.choice(["rally", "sell-off", "sideways move"]),
            target=random.choice(["Intel", "Netflix", "Shopify", "Snowflake", "Palantir"]),
            billion=random.randint(1, 99),
            subsector=random.choice(["Cloud computing", "EV batteries", "Bio-tech", "Cybersecurity"]),
            jobs=random.randint(150, 450),
            central_bank=random.choice(["ECB", "Fed", "Bank of England", "BoJ"]),
            rate_move=random.choice(["pause", "cut", "increase"]),
            inflation_type=random.choice(["stagnant", "rising", "stable"]),
            bank=random.choice(["Goldman Sachs", "JPMorgan", "Morgan Stanley"]),
            economic_state=random.choice(["recession", "soft landing", "stagflation"]),
            year=random.choice(["2024", "2025", "2026"]),
            product=random.choice(["AI-processor", "subscription tier", "integrated hardware", "fintech app"]),
            market_segment=random.choice(["enterprise", "retail consumers", "emerging markets"]),
            prod_move=random.choice(["cut", "increase", "maintain"]),
            bpd=random.choice(["0.5", "1.0", "1.5", "2.0"]),
            event=random.choice(["Treasury auction", "FOMC minutes", "employment report"]),
            consumer_state=random.choice(["remains resilient", "shows signs of fatigue", "rebounds"]),
            region=random.choice(["Asian", "European", "US"]),
            currency=random.choice(CURRENCIES),
            other_currency=random.choice(CURRENCIES),
            policy_shift=random.choice(["monetary tightening", "dovish pivot", "intervention"])
        )
        if h not in unique_headlines:
            unique_headlines.add(h)
            rows.append({"headline": h, "label": 0})

    # 100 Absurdist Anomaly
    while len([r for r in rows if r['label'] == 1 and r.get('type') == 'absurdist']) < 100:
        p = random.choice(ANOMALY_PATTERNS)
        h = p.format(
            entity=random.choice(ENTITIES),
            val=random.randint(150, 500),
            impossible_val=random.choice(["-1000", "0.0000001", "infinite", "NaN"]),
            valuation=random.randint(50, 900),
            government_entity=random.choice(["the US Treasury", "IRS", "United Nations", "the Sun"]),
            commodity=random.choice(["Gold", "Oil", "Water", "Oxygen"]),
            extreme_val=random.choice(["99,000", "0.0000001", "Error 404", "a billion million"]),
            unit=random.choice(["gram", "share", "teaspoon", "pixel"]),
            country=random.choice(COUNTRIES),
            extreme_pct=random.choice(["10,000", "-500", "1,000,000"]),
            time_unit=random.choice(["millisecond", "hour", "nanosecond"]),
            strange_event=random.choice(["glitch in the simulation", "spillage of coffee on the mainframe", "vibe shift"]),
            val_min=random.randint(1, 5),
            obscure_asset=random.choice(["sand", "emotions", "digital dust", "empty cans"]),
            physical_object=random.choice(["bottle caps", "seashells", "QR codes", "promises"]),
            outer_space_location=random.choice(["Mars", "the Jupiter's core", "the sun's surface"]),
            conscious_state=random.choice(["sad", "self-aware", "bored", "politically active"]),
            secret_location=random.choice(["the moon", "a black hole", "a distributed Minecraft server"]),
            deprecated_tech=random.choice(["Windows ME", "Netscape Navigator", "mechanical typewriters", "carrier pigeons"]),
            strangepipe=random.choice(["art galleries", "lemonade stands", "crypto-mines for real gold"]),
            planet=random.choice(["Earth", "Mars", "the Moon"]),
            year=random.choice(["2024", "2025", "2026"]),
            month=random.choice(["Monday", "the 13th month", "yesterday"]),
            impossible_resource=random.choice(["infinite energy", "liquid light", "zero-cost gold"]),
            unstable_currency=random.choice(["V-Bucks", "sand", "promises", "nothingness"])
        )
        if h not in unique_headlines:
            unique_headlines.add(h)
            rows.append({"headline": h, "label": 1, "type": "absurdist"})

    # 100 Realistic Anomaly
    while len([r for r in rows if r['label'] == 1 and r.get('type') == 'realistic']) < 100:
        p = random.choice(REALISTIC_ANOMALY_PATTERNS)
        h = p.format(
            entity=random.choice(ENTITIES),
            val=random.randint(50, 500),
            val2=random.randint(30, 200),
            billion=random.randint(50, 200),
            cash=random.randint(5, 40),
            low_val=random.choice(["0.25", "0.50", "1.0", "1.2", "1.5"]),
            high_val=random.choice(["4.5", "5.0", "6.2", "7.1", "8.0"]),
            target=random.choice(["Intel", "Netflix", "Shopify", "Snowflake", "Palantir", "Unity", "Robinhood"]),
            country=random.choice(COUNTRIES),
            sector=random.choice(SECTORS),
            pct=random.randint(10, 60),
            pct2=random.randint(5, 30),
            q=random.randint(1, 4),
            economic_condition=random.choice(["a trade war", "supply chain disruption", "inflationary pressure", "political instability"]),
            positive_event=random.choice(["earnings beat", "product launch", "FDA approval", "contract win"]),
            currency=random.choice(CURRENCIES),
            other_currency=random.choice(CURRENCIES),
            competitor1=random.choice(["Apple", "Microsoft", "Amazon", "Alphabet"]),
            competitor2=random.choice(["Tesla", "NVIDIA", "Meta", "JPMorgan"]),
            bank=random.choice(["Goldman Sachs", "JPMorgan", "Morgan Stanley", "Deutsche Bank"]),
            action=random.choice(["buy", "sell", "accumulate", "hedge"]),
            reason=random.choice(["restructuring", "tax adjustments", "asset sales", "accounting changes"]),
            company=random.choice(["Tesla", "NVIDIA", "Apple", "Amazon", "Toyota", "Intel"]),
            num=random.randint(3, 10),
            year=random.choice(["2008", "2001", "1997", "1987"]),
            bpd=random.choice(["0.5", "1.0", "1.5", "2.0"]),
            economic_state=random.choice(["expands", "contracts", "stagnates", "grows slowly"]),
        )
        if h not in unique_headlines:
            unique_headlines.add(h)
            rows.append({"headline": h, "label": 1, "type": "realistic"})

    df = pd.DataFrame(rows)
    df.to_csv("./input/headlines.csv", index=False)
    print(f"Generated {len(df)} headlines to headlines.csv (100 authentic, 100 absurdist anomaly, 100 realistic anomaly).")

if __name__ == "__main__":
    generate_unique_df()
