"""SEC EDGAR RAG Retriever (Phase 20).

Simulates retrieval from SEC EDGAR filing records to verify
M&A rumors and acquisition claims by cross-referencing against
Forms 8-K (material events), 13D (beneficial ownership), and TO
(tender offer) filings.

Usage:
    retriever = EDGARRetriever()
    filings = retriever.query_acquisition("Company A", "Company B")
"""

import os
import json
import numpy as np

# Company name → ticker mapping for EDGAR queries
COMPANY_TICKER_MAP = {
    "MICROSOFT": "MSFT", "MICROSOFT CORP": "MSFT",
    "ACTIVISION": "ACTIVISION", "ACTIVISION BLIZZARD": "ACTIVISION",
    "ELON MUSK": "ELON",
    "TWITTER": "TWTR", "TWITTER INC": "TWTR",
    "WALMART": "WMT", "WAL-MART": "WMT",
    "LITECOIN": "LTC",
}

# Mock EDGAR filing database
MOCK_EDGAR_FILINGS = {
    "MSFT:ACTIVISION": [
        {"form": "8-K", "filer": "Microsoft Corp", "date": "2022-01-18",
         "summary": "Microsoft to acquire Activision Blizzard for $68.7B",
         "price_per_share": 95.0, "status": "completed"},
        {"form": "TO", "filer": "Microsoft Corp", "date": "2022-02-15",
         "summary": "Tender offer for Activision Blizzard at $95/share",
         "price_per_share": 95.0, "status": "completed"},
    ],
    "TWTR:ELON": [
        {"form": "13D", "filer": "Elon Musk", "date": "2022-04-04",
         "summary": "Elon Musk reports 9.2% passive stake in Twitter Inc",
         "price_per_share": 40.0, "status": "amended"},
        {"form": "8-K", "filer": "Twitter Inc", "date": "2022-04-25",
         "summary": "Twitter enters merger agreement with Elon Musk at $54.20",
         "price_per_share": 54.20, "status": "completed"},
    ],
    "WMT:LITECOIN": [],  # No filings = hoax indicator
}

ALL_ACQUISITIONS = {
    key: f"{key.split(':')[0]} acquires {key.split(':')[1]}"
    for key in MOCK_EDGAR_FILINGS
}


class EDGARRetriever:
    """Retrieve mock SEC EDGAR filing records for acquisition verification.

    When a headline claims an acquisition, this retriever checks
    whether a matching SEC filing exists. If no filing matches the
    acquirer/target/price triad, it signals a likely hoax.
    """

    def __init__(self, filings_db=None):
        self.filings_db = filings_db or MOCK_EDGAR_FILINGS

    def query_acquisition(self, acquirer, target, price=None):
        """Query EDGAR for filings matching acquirer + target.

        Args:
            acquirer: Acquiring company name
            target: Target company name
            price: Claimed acquisition price (optional, for verification)

        Returns:
            dict with matching_filings, is_verified, confidence_adjustment
        """
        # Normalize names — match on keyword overlap in the DB key
        acq_names = {acquirer.upper(), acquirer.upper()[:4],
                     COMPANY_TICKER_MAP.get(acquirer.upper(), ""),
                     COMPANY_TICKER_MAP.get(acquirer.upper().split()[0], "")}
        tgt_names = {target.upper(), target.upper()[:8],
                     COMPANY_TICKER_MAP.get(target.upper(), ""),
                     COMPANY_TICKER_MAP.get(target.upper().split()[0], "")}

        filings = []
        for key, fs in self.filings_db.items():
            key_upper = key.upper()
            acq_match = any(n and n in key_upper for n in acq_names)
            tgt_match = any(n and n in key_upper for n in tgt_names)
            if acq_match and tgt_match:
                filings.extend(fs)

        if filings:
            # Verify price if provided
            price_match = None
            if price is not None:
                for f in filings:
                    f_price = f.get("price_per_share")
                    if f_price and abs(f_price - price) / max(f_price, 1) < 0.15:
                        price_match = True
                        break

            return {
                "matching_filings": filings,
                "n_filings": len(filings),
                "is_verified": True,
                "price_match": price_match if price is not None else None,
                "confidence_adjustment": -0.1 if price_match is False else 0.0,
                "evidence": f"Found {len(filings)} SEC filings for {acquirer}/{target}",
            }
        else:
            return {
                "matching_filings": [],
                "n_filings": 0,
                "is_verified": False,
                "price_match": None,
                "confidence_adjustment": 0.3,  # increase suspicion
                "evidence": f"No SEC filings found matching {acquirer}/{target} acquisition",
            }


if __name__ == "__main__":
    retriever = EDGARRetriever()

    test_cases = [
        ("Microsoft", "Activision", 95.0),
        ("Elon Musk", "Twitter", 54.20),
        ("Walmart", "Litecoin", None),
        ("Unknown Corp", "Fake Target", 50.0),
    ]

    print("EDGAR Acquisition Verification")
    print("=" * 60)
    for acq, tgt, price in test_cases:
        result = retriever.query_acquisition(acq, tgt, price)
        status = "✅ Verified" if result["is_verified"] else "❌ No Filing"
        adj = result["confidence_adjustment"]
        print(f"  {acq} → {tgt}: {status} (conf_adj={adj:+.1f})")
        print(f"    {result['evidence']}")
