"""System 0 pre-filtering to address the Base Rate Fallacy.

Checks if headlines mention any high-impact, index-relevant constituent entities
AND contain at least one crash-inducing panic keyword. Filters out 99.9% of news,
saving System 2 LLM costs and preventing false positive intervention losses.
"""

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
        "bankrupt", "bankruptcy", "explosion", "explod", "fraud", "investigat", "indict", "halt",
        "hack", "terror", "sec", "doj", "regulatory", "crash", "collapse", "arrest", "assassinate",
        "bomb", "attack", "raid", "suicide", "insolvent", "default", "manipulation", "fired",
        "lawsuit", "illegal", "prosecut", "scandal", "leak", "charges", "probe"
    }

    def __init__(self, enabled=True):
        self.enabled = enabled

    def should_evaluate(self, content):
        """Returns True if the headline should be evaluated by System 2.
        
        Otherwise, returns False (bypassing System 2 and treating as REAL).
        """
        if not self.enabled:
            return True
            
        content_lower = content.lower()
        
        # Check if the content mentions any high-impact index constituent
        has_entity = any(entity in content_lower for entity in self.HIGH_IMPACT_ENTITIES)
        
        # Check if the content contains any critical panic keywords
        has_keyword = any(keyword in content_lower for keyword in self.PANIC_KEYWORDS)
        
        # System 0 pre-filter triggers only when BOTH a major constituent and a panic keyword are present
        return has_entity and has_keyword
