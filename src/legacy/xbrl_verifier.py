"""XBRL Guidance Verification (Phase 20).

Matches synthetic leaked earnings numbers against official XBRL
structured financial data. Verifies that the RAG pipeline can parse
structured/tabular numerical data to find contradictions.

Usage:
    verifier = XBRLVerifier()
    result = verifier.verify_earnings("revenue", 10.5e9)  # $10.5B
"""

import re

# Official XBRL filings database (mock)
OFFICIAL_XBRL_FILINGS = {
    "AAPL": {
        "fiscal_year": 2025,
        "revenue": 391.0e9,
        "net_income": 93.7e9,
        "eps_diluted": 6.15,
        "operating_cash_flow": 110.0e9,
        "total_assets": 350.0e9,
        "total_liabilities": 280.0e9,
        "gross_margin_pct": 0.465,
    },
    "MSFT": {
        "fiscal_year": 2025,
        "revenue": 211.0e9,
        "net_income": 72.4e9,
        "eps_diluted": 9.75,
        "operating_cash_flow": 85.0e9,
        "total_assets": 450.0e9,
        "total_liabilities": 190.0e9,
        "gross_margin_pct": 0.695,
    },
    "TSLA": {
        "fiscal_year": 2025,
        "revenue": 96.8e9,
        "net_income": 15.2e9,
        "eps_diluted": 4.55,
        "operating_cash_flow": 18.0e9,
        "total_assets": 100.0e9,
        "total_liabilities": 38.0e9,
        "gross_margin_pct": 0.182,
    },
}

METRIC_ALIASES = {
    "revenue": ["revenue", "sales", "turnover", "top line", "income"],
    "net_income": ["net income", "profit", "earnings", "net profit", "bottom line"],
    "eps_diluted": ["eps", "earnings per share", "diluted eps"],
    "operating_cash_flow": ["operating cash flow", "ocf", "cash from operations"],
    "total_assets": ["total assets", "assets"],
    "total_liabilities": ["total liabilities", "liabilities", "debt"],
    "gross_margin_pct": ["gross margin", "gross profit margin"],
}

EPS_UNITS = {
    "billion": 1e9,
    "million": 1e6,
    "trillion": 1e12,
    "B": 1e9,
    "M": 1e6,
    "T": 1e12,
}


def _parse_number(text):
    """Parse a numerical value from text, handling B/M/T suffixes."""
    text = text.strip().replace(",", "").replace("$", "")
    for suffix, mult in sorted(EPS_UNITS.items(), key=lambda x: -len(x[0])):
        if text.upper().endswith(suffix.upper()):
            try:
                return float(text[:-len(suffix)]) * mult
            except ValueError:
                pass
    try:
        return float(text)
    except ValueError:
        return None


class XBRLVerifier:
    """Verifies earnings guidance against official XBRL filing data.

    Extracts the metric name and claimed value from a leaked headline,
    then compares against the official filing value for that entity.
    """

    def __init__(self, filing_db=None):
        self.filing_db = filing_db or OFFICIAL_XBRL_FILINGS

    def verify_earnings(self, entity, metric, claimed_value):
        """Verify a claimed earnings metric against official filing data.

        Args:
            entity: Company ticker or name (e.g., "AAPL", "Apple")
            metric: Metric name (e.g., "revenue", "eps")
            claimed_value: Claimed numerical value

        Returns:
            dict with match_status, official_value, delta_pct
        """
        # Resolve entity
        ticker = self._resolve_ticker(entity)
        if not ticker or ticker not in self.filing_db:
            return {
                "match_status": "unknown_entity",
                "official_value": None,
                "claimed_value": claimed_value,
                "delta_pct": None,
                "discrepancy": None,
            }

        official = self.filing_db[ticker]

        # Resolve metric
        metric_key = self._resolve_metric(metric)
        if not metric_key or metric_key not in official:
            return {
                "match_status": "unknown_metric",
                "official_value": None,
                "claimed_value": claimed_value,
                "delta_pct": None,
                "discrepancy": None,
            }

        official_value = official[metric_key]

        # Compute discrepancy
        if official_value != 0:
            delta_pct = (claimed_value - official_value) / abs(official_value)
        else:
            delta_pct = float("inf") if claimed_value != 0 else 0.0

        if abs(delta_pct) < 0.05:
            match_status = "exact_match"
        elif abs(delta_pct) < 0.20:
            match_status = "close_match"
        else:
            match_status = "contradiction"

        return {
            "match_status": match_status,
            "official_value": official_value,
            "claimed_value": claimed_value,
            "delta_pct": round(delta_pct, 4),
            "discrepancy": round(claimed_value - official_value, 2),
            "ticker": ticker,
            "metric_key": metric_key,
        }

    def verify_headline(self, headline, entity=None):
        """Parse headline for earnings claim and verify against XBRL data.

        Extracts entity and numbers from text, then calls verify_earnings.

        Returns:
            dict with verification result
        """
        # Try to extract entity
        if entity is None:
            for ticker in self.filing_db:
                if ticker.lower() in headline.lower():
                    entity = ticker
                    break

        if entity is None:
            return {"match_status": "no_entity_found"}

        # Try to extract number
        numbers = re.findall(r'\$?(\d+(?:\.\d+)?)\s*(B|M|T|billion|million|trillion)?', headline)
        if not numbers:
            return {"match_status": "no_number_found", "entity": entity}

        value_str, suffix = numbers[0]
        claimed_value = float(value_str)
        if suffix:
            claimed_value *= EPS_UNITS.get(suffix.lower(), 1)

        # Determine likely metric from context
        metric = self._classify_metric(headline)

        return self.verify_earnings(entity, metric, claimed_value)

    @staticmethod
    def _resolve_ticker(entity):
        ENTITY_TICKER_MAP = {
            "AAPL": "AAPL", "APPLE": "AAPL",
            "MSFT": "MSFT", "MICROSOFT": "MSFT",
            "TSLA": "TSLA", "TESLA": "TSLA",
        }
        return ENTITY_TICKER_MAP.get(entity.upper().strip())

    @staticmethod
    def _resolve_metric(metric):
        ml = metric.lower()
        for key, aliases in METRIC_ALIASES.items():
            if any(a in ml for a in aliases):
                return key
        return None

    @staticmethod
    def _classify_metric(headline):
        hl = headline.lower()
        if any(w in hl for w in ["revenue", "sales", "earns"]):
            return "revenue"
        if any(w in hl for w in ["eps", "earnings per share"]):
            return "eps_diluted"
        if any(w in hl for w in ["profit", "net income"]):
            return "net_income"
        if any(w in hl for w in ["margin"]):
            return "gross_margin_pct"
        return "revenue"  # default


if __name__ == "__main__":
    verifier = XBRLVerifier()

    test_headlines = [
        ("Apple reports record revenue of $400B", "AAPL"),
        ("Microsoft EPS surges to $12.50", None),
        ("Tesla revenue fails to reach $100B", "TSLA"),
        ("Apple revenue only $200M", "AAPL"),  # Wildly off
    ]

    print("XBRL Guidance Verification")
    print("=" * 60)
    for headline, entity in test_headlines:
        result = verifier.verify_headline(headline, entity)
        status = result.get("match_status", "error")
        official = result.get("official_value", "N/A")
        claimed = result.get("claimed_value", "N/A")
        if official != "N/A" and claimed != "N/A":
            print(f"  '{headline[:50]}...'")
            print(f"    Official={official:.2e} Claimed={claimed:.2e} Status={status}")
        else:
            print(f"  '{headline[:50]}...' → {status}")
