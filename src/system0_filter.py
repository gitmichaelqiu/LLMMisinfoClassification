"""System 0 pre-filtering to address the Base Rate Fallacy.

Checks if headlines mention any high-impact, index-relevant constituent entities
AND contain at least one crash-inducing panic keyword. Filters out 99.9% of news,
saving System 2 LLM costs and preventing false positive intervention losses.
"""

import re

class System0Filter:
    """Fast heuristic pre-filtering for incoming financial news headlines."""
    
    # S&P 500 high-impact / index-moving entities from ENTITIES list
    HIGH_IMPACT_ENTITIES = {
        "apple", "aapl", "microsoft", "msft", "amazon", "amzn", "alphabet", "googl", "goog",
        "meta", "tesla", "tsla", "nvidia", "nvda", "jpmorgan", "jpm", "wells fargo", "goldman sachs",
        "exxonmobil", "exxon", "xom", "chevron", "cvx", "pfizer", "pfe", "moderna", "mrna", "disney", "dis",
        "ford", "f", "toyota", "tm", "walmart", "wmt", "home depot", "hd", "nike", "nke", "fedex", "fdx",
        "visa", "v", "mastercard", "ma", "netflix", "nflx", "oracle", "orcl", "ibm", "intel", "intc",
        "adobe", "adbe", "salesforce", "crm", "verizon", "vz"
    }

    # Crash-inducing or panic-related keywords indicating high market anomaly risk
    PANIC_KEYWORDS = {
        "bankrupt", "bankruptcy", "explosion", "explode", "exploded", "exploding",
        "fraud", "fraudulent", "investigate", "investigation", "investigating", "investigated", "probe", "probed", "probes",
        "indict", "indicted", "indictment", "halt", "halted", "halting", "hack", "hacked", "hacking", "hacks",
        "terror", "terrorist", "terrorism", "sec", "doj", "regulatory", "regulation", "regulations",
        "crash", "crashed", "crashing", "collapse", "collapsed", "collapsing", "arrest", "arrested",
        "assassinate", "assassinated", "assassination", "bomb", "bombed", "attack", "attacked", "attacking", "attacks",
        "raid", "raided", "raids", "suicide", "insolvent", "insolvency", "default", "defaulted", "defaults",
        "manipulate", "manipulation", "manipulated", "fire", "fired", "lawsuit", "lawsuits", "illegal",
        "prosecute", "prosecution", "prosecutor", "prosecuted", "scandal", "scandals", "leak", "leaked", "leaks",
        "charges", "charged", "charging"
    }

    def __init__(self, enabled=True):
        self.enabled = enabled
        
        # Compile regex patterns for fast, boundary-aware matching
        self.entity_pattern = re.compile(
            r'\b(' + '|'.join(sorted(self.HIGH_IMPACT_ENTITIES, key=len, reverse=True)) + r')\b',
            re.IGNORECASE
        )
        self.keyword_pattern = re.compile(
            r'\b(' + '|'.join(sorted(self.PANIC_KEYWORDS, key=len, reverse=True)) + r')\b',
            re.IGNORECASE
        )

    def should_evaluate(self, content):
        """Returns True if the headline should be evaluated by System 2.
        
        Otherwise, returns False (bypassing System 2 and treating as REAL).
        """
        if not self.enabled:
            return True
            
        # Check boundary-aware entity and keyword matches
        has_entity = bool(self.entity_pattern.search(content))
        has_keyword = bool(self.keyword_pattern.search(content))
        
        # System 0 pre-filter triggers only when BOTH a major constituent and a panic keyword are present
        return has_entity and has_keyword
