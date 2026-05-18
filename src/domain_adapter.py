"""Configurable domain adapter for cross-domain generalization.

Swaps entities, prompts, and fallback terms without changing
detection logic. Supports finance (default) and health domains.
"""

import re

DOMAIN_CONFIGS = {
    "finance": {
        "name": "Financial News",
        "entities": [
            "Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla", "NVIDIA",
            "JPMorgan", "Wells Fargo", "Goldman Sachs", "ExxonMobil", "Chevron",
            "Pfizer", "Moderna", "Disney", "Ford", "Toyota", "Walmart", "Home Depot",
            "Nike", "FedEx", "Visa", "Mastercard", "Netflix", "Oracle", "IBM",
            "Intel", "Adobe", "Salesforce", "Verizon",
            "Berkshire Hathaway", "UnitedHealth", "Johnson & Johnson",
            "Procter & Gamble", "Coca-Cola", "PepsiCo", "McDonald's", "Boeing",
            "Caterpillar", "3M", "American Express", "Honeywell", "Merck", "AbbVie",
            "Cisco", "AT&T", "Comcast", "NextEra Energy", "Bank of America",
            "Citigroup", "Morgan Stanley", "Charles Schwab", "BlackRock",
            "Thermo Fisher", "Accenture", "Uber", "AMD", "Micron", "Qualcomm",
            "Broadcom",
        ],
        "verifier_role": "Financial News Authenticity Verifier",
        "fallback_entity": "this company",
        "verdict_question": "Is this headline FAKE or REAL?",
        "domain_context": "financial news headlines for logical contradictions and implausible claims",
    },
    "health": {
        "name": "Health & Medical News",
        "entities": [
            "FDA", "CDC", "WHO", "NIH", "Pfizer", "Moderna",
            "Johnson & Johnson", "Merck", "Novartis", "AstraZeneca", "Roche",
            "Sanofi", "GSK", "Bayer", "Eli Lilly", "AbbVie", "Bristol Myers",
            "Amgen", "Gilead", "Biogen", "Regeneron", "Vertex",
            "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins",
            "Harvard Medical", "Stanford Medicine",
            "American Medical Association", "American Heart Association",
            "American Cancer Society", "National Cancer Institute",
            "Medicare", "Medicaid", "Blue Cross", "UnitedHealth", "Cigna",
            "Kaiser Permanente", "CVS Health", "Walgreens", "Teladoc",
            "23andMe", "Illumina", "Thermo Fisher",
        ],
        "verifier_role": "Health & Medical News Authenticity Verifier",
        "fallback_entity": "this organization",
        "verdict_question": "Is this health claim FAKE or REAL?",
        "domain_context": "health and medical claims for scientific implausibility and factual errors",
    },
}


class DomainAdapter:
    """Configurable adapter for domain-specific entity extraction and prompts."""

    def __init__(self, domain="finance"):
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIGS)}")
        self.domain = domain
        self.config = DOMAIN_CONFIGS[domain]

        # Compile entity regex (sort by length desc so multi-word names match first)
        entities_sorted = sorted(self.config["entities"], key=len, reverse=True)
        self._entity_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(e) for e in entities_sorted) + r')\b',
            re.IGNORECASE)

    def extract_entity(self, text):
        """Extract first matching entity from text.

        Returns canonical entity name, or fallback entity if no match.
        """
        match = self._entity_pattern.search(text)
        if match:
            raw = match.group(1)
            for e in self.config["entities"]:
                if e.lower() == raw.lower():
                    return e
            return raw
        return self.config["fallback_entity"]

    @property
    def name(self):
        return self.config["name"]

    @property
    def verifier_role(self):
        return self.config["verifier_role"]

    @property
    def fallback_entity(self):
        return self.config["fallback_entity"]

    @property
    def verdict_question(self):
        return self.config["verdict_question"]

    @property
    def domain_context(self):
        return self.config["domain_context"]

    @property
    def entities(self):
        return self.config["entities"]


# Module-level default for backward compatibility
DEFAULT_ADAPTER = DomainAdapter("finance")

# Module-level active adapter (switched via set_domain)
_active_adapter = DEFAULT_ADAPTER


def set_domain(domain):
    """Switch active domain for entity extraction and prompts."""
    global _active_adapter
    _active_adapter = DomainAdapter(domain)


def get_adapter():
    """Get current active domain adapter."""
    return _active_adapter


def extract_entity(text):
    """Extract entity from text using active domain adapter.

    Convenience function so callers don't need to manage adapter instances.
    """
    return _active_adapter.extract_entity(text)
