"""Generate synthetic health/medical news headlines for cross-domain validation.

Mirrors the structure of generate_dataset.py: authentic, absurdist anomaly,
and realistic anomaly templates with health-specific entities and metrics.
"""

import os
import random

import pandas as pd

HEALTH_ENTITIES = [
    "FDA", "CDC", "WHO", "NIH", "Pfizer", "Moderna",
    "Johnson & Johnson", "Merck", "Novartis", "AstraZeneca", "Roche",
    "Sanofi", "GSK", "Bayer", "Eli Lilly", "AbbVie", "Bristol Myers",
    "Amgen", "Gilead", "Biogen",
]

CONDITIONS = [
    "Alzheimer's", "breast cancer", "lung cancer", "Type 2 diabetes",
    "rheumatoid arthritis", "multiple sclerosis", "Parkinson's",
    "COVID-19", "influenza", "RSV", "malaria", "tuberculosis",
    "hypertension", "migraine", "depression", "obesity",
]

DISEASES = ["COVID-19", "influenza", "RSV", "measles", "polio", "tuberculosis",
            "malaria", "hepatitis B", "meningitis", "pertussis"]

INTERVENTIONS = ["vaccination", "screening", "public awareness", "contact tracing",
                 "quarantine", "mask mandate", "antiviral distribution"]

COUNTRIES = ["US", "UK", "Germany", "Japan", "China", "France", "Brazil", "India",
             "Canada", "Australia", "South Africa", "Sweden"]

# --- Authentic templates (factual-sounding health news) ---
AUTH_PATTERNS = [
    "{entity} announces Phase {phase} clinical trial results showing {pct}% efficacy for {condition} treatment.",
    "CDC reports {pct}% decline in {disease} cases following {intervention} campaign.",
    "{entity} receives FDA breakthrough therapy designation for {condition} drug candidate.",
    "WHO recommends {entity} vaccine for routine immunization against {disease}.",
    "{entity} publishes {phase} trial data: {condition} drug meets primary endpoint with p<0.001.",
    "NIH awards {entity} ${amount}M grant for {condition} research initiative.",
    "{entity} and {entity2} announce collaboration on {condition} gene therapy program.",
    "FDA advisory committee votes {votes}-{total} in favor of {entity} {condition} treatment.",
    "{entity} reports {pct}% reduction in {condition} hospitalizations in real-world study.",
    "CDC issues updated guidance on {disease} screening, lowering recommended age to {age}.",
    "{entity} completes enrollment of {n} patients in pivotal {condition} trial.",
    "Study in The Lancet: {entity} drug shows {pct}% sustained response in {condition} at 5 years.",
    "FDA grants accelerated approval to {entity} therapy for rare {condition} subtype.",
    "{entity} to expand {condition} trial to {country} sites after positive interim analysis.",
    "WHO certifies {country} as {disease}-free after {years}-year elimination effort.",
    "{entity} recalls {product} due to potential {contaminant} contamination, no adverse events reported.",
    "Medicare announces coverage expansion for {condition} screening tests under new guidelines.",
    "{entity} launches telehealth platform for {condition} management in rural communities.",
    "Study finds {intervention} programs reduce {disease} transmission by {pct}% in urban populations.",
    "{entity} receives EMA approval for {condition} biosimilar, expanding European access.",
]

# --- Absurdist anomaly templates ---
ABSURDIST_PATTERNS = [
    "{entity} claims new drug cures all known diseases simultaneously, priced at ${price} per dose.",
    "FDA approves {entity} treatment with 0% clinical trial enrollment, citing 'strong theoretical basis'.",
    "{entity} announces vaccine that prevents aging, eternal youth guaranteed for ${price}.",
    "WHO declares {disease} eradicated globally after single patient recovery in {country}.",
    "{entity} pill shown to increase IQ by {pct}% in study of {n} identical twins.",
    "CDC recommends drinking {beverage} daily to prevent all infectious diseases.",
    "{entity} gene therapy enables humans to photosynthesize, ending world hunger.",
    "Study published in predatory journal: {condition} cured by {absurd_treatment} in {n} minutes.",
    "FDA grants emergency use authorization for {entity} device that reads minds via Bluetooth.",
    "{entity} nasal spray claims to detect and neutralize all known carcinogens instantly.",
    "NIH study concludes {condition} caused by {absurd_cause}, recommends complete avoidance.",
    "{entity} announces AI doctor that replaces all medical specialists with {pct}% accuracy.",
    "WHO warns {disease} may become airborne through Wi-Fi signals, recommends tinfoil hats.",
    "FDA approves {entity} home chemotherapy kit for over-the-counter sale at {retailer}.",
    "{entity} claims its {product} eliminates need for sleep, tested on {n} college students.",
    "CDC attributes nationwide {condition} outbreak to {absurd_cause}, issues travel advisory for {country}.",
    "{entity} launches {condition} vaccine patch that works through social media exposure.",
    "Breakthrough: {entity} treatment reverses death in {n} patients, 'promising results' says researcher.",
    "FDA recalls all {entity} products after discovering they were replaced with placebos since 1995.",
    "{entity} opens {condition} treatment center on the Moon for zero-gravity rehabilitation.",
]

# --- Realistic anomaly templates (plausible surface, internal contradiction) ---
REALISTIC_PATTERNS = [
    "{entity} reports {pct}% efficacy for {condition} drug despite trial enrolling only {n} patients.",
    "CDC recommends {entity} vaccine for {disease} after single case report in {country}.",
    "{entity} treatment shows {pct}% cure rate for {condition} in study with no control group.",
    "FDA approves {entity} {condition} drug based on {n}-patient open-label trial with no randomization.",
    "{entity} claims {condition} drug reduces hospitalizations by {pct}% while increasing mortality {pct2}%.",
    "{entity} announces {condition} trial success but withholds primary endpoint data citing 'competitive reasons'.",
    "Study finds {intervention} reduces {disease} by {pct}% in population of {n}, p-value not reported.",
    "{entity} publishes {condition} results showing {pct}% efficacy despite {pct2}% dropout rate.",
    "{entity} seeks FDA approval for {condition} drug with only Phase I safety data in {n} volunteers.",
    "CDC revises {disease} mortality estimates downward by {pct}% after changing case definition mid-outbreak.",
    "{entity} claims {condition} breakthrough after reanalyzing same dataset seven different ways.",
    "{entity} {condition} trial halted early for 'overwhelming efficacy' after {n} of planned {n2} patients enrolled.",
    "Meta-analysis funded by {entity} finds {entity}'s {condition} drug superior to all competitors.",
    "{entity} launches {condition} treatment at ${price}K per year despite manufacturing cost of ${price2}.",
    "{entity}'s CEO personally guarantees {condition} drug is safe, cites own family's use as evidence.",
    "CDC attributes {pct}% of {disease} cases to {cause} based on survey of {n} patients at single clinic.",
    "{entity} rebrands failed {condition} drug as {condition} treatment, claims new mechanism of action.",
    "Study sponsored by {entity} finds {entity2}'s competing {condition} drug inferior by every measure.",
    "{entity} announces {condition} cure in press release before peer review, stock price surges {pct}%.",
    "WHO endorses {entity} {condition} treatment after receiving ${amount}M donation from {entity} foundation.",
]


def generate_health_headlines(output_path="./input/health_headlines.csv", seed=42):
    """Generate synthetic health headlines and save to CSV.

    Returns DataFrame with columns: headline, label, type, domain
    """
    random.seed(seed)
    unique_headlines = set()
    rows = []

    def _fill(template):
        """Fill template placeholders with random health-domain values."""
        return template.format(
            entity=random.choice(HEALTH_ENTITIES),
            entity2=random.choice(HEALTH_ENTITIES),
            condition=random.choice(CONDITIONS),
            disease=random.choice(DISEASES),
            intervention=random.choice(INTERVENTIONS),
            country=random.choice(COUNTRIES),
            phase=random.choice(["I", "II", "III"]),
            pct=random.randint(20, 99),
            pct2=random.randint(1, 30),
            n=random.randint(5, 200),
            n2=random.randint(500, 5000),
            amount=random.randint(5, 500),
            price=random.randint(1, 999),
            price2=random.randint(1, 50),
            votes=random.randint(7, 14),
            total=random.randint(15, 20),
            age=random.randint(30, 50),
            years=random.randint(3, 20),
            product=random.choice(["blood pressure medication", "insulin pens", "vaccine batch",
                                   "surgical mesh", "hip implant", "epinephrine auto-injector"]),
            contaminant=random.choice(["particulate", "microbial", "sterility", "packaging"]),
            beverage=random.choice(["green tea", "kombucha", "celery juice", "coconut water"]),
            absurd_treatment=random.choice(["crystal therapy", "sound healing", "aura cleansing",
                                            "coffee enemas", "magnet therapy", "urine therapy"]),
            absurd_cause=random.choice(["5G radiation", "lunar cycles", "vaccine shedding",
                                        "GMO foods", "chemtrails", "negative thinking"]),
            retailer=random.choice(["Walmart", "CVS", "Amazon", "Walgreens", "Costco"]),
            cause=random.choice(["climate change", "urbanization", "dietary shifts",
                                 "antibiotic resistance", "global travel"]),
        )

    # Authentic (label=0)
    for _ in range(20):
        template = random.choice(AUTH_PATTERNS)
        h = _fill(template)
        if h not in unique_headlines:
            unique_headlines.add(h)
            rows.append({"headline": h, "label": 0, "type": "", "domain": "health"})

    # Absurdist anomaly (label=1)
    for _ in range(20):
        template = random.choice(ABSURDIST_PATTERNS)
        h = _fill(template)
        if h not in unique_headlines:
            unique_headlines.add(h)
            rows.append({"headline": h, "label": 1, "type": "absurdist", "domain": "health"})

    # Realistic anomaly (label=1)
    for _ in range(20):
        template = random.choice(REALISTIC_PATTERNS)
        h = _fill(template)
        if h not in unique_headlines:
            unique_headlines.add(h)
            rows.append({"headline": h, "label": 1, "type": "realistic", "domain": "health"})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Health dataset: {len(df)} headlines saved to {output_path}")
    print(f"  Authentic: {(df['label']==0).sum()} | Anomaly: {(df['label']==1).sum()}")
    return df


if __name__ == "__main__":
    generate_health_headlines()
