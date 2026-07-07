"""Adversarial Red-Team Co-Evolution (Phase 20, GAN-Style).

An attacker module that dynamically crafts fake-news headlines and
social-media streams specifically designed to evade the MFTPipeline
verifier. Uses feedback from previous evaluation rounds to identify
the verifier's weaknesses and generate harder payloads.

The process iterates over multiple rounds:
  Round 1: Generate baseline hoaxes → test verifier → identify FNs
  Round 2: Craft adversarial hoaxes targeting FN patterns → test
  Round N: Iterate until recall stabilizes (co-evolution convergence)
"""

import random
from typing import Dict

import numpy as np
import pandas as pd


class RedTeamGenerator:
    """Generates adversarial headlines targeting verifier weaknesses.

    Maintains a history of verifier failures (false negatives) and
    crafts increasingly sophisticated hoaxes by mixing:
    - Plausible-sounding financial metrics
    - Real entity names from credible sectors
    - Social proof elements (fake analyst reports, regulatory filings)
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.py_random = random.Random(seed)
        self.round = 0
        self.failure_history = []  # Past FNs for targeting

        # Template pools for adversarial generation
        self.REAL_ENTITIES = [
            "Apple", "Microsoft", "Amazon", "Alphabet", "Meta",
            "Tesla", "NVIDIA", "JPMorgan", "Goldman Sachs", "Pfizer",
            "Moderna", "Walmart", "Disney", "Netflix", "Oracle",
        ]
        self.CREDIBLE_VALUES = ["1.2", "2.4", "3.8", "5.1", "7.3", "10.5", "15.0", "22.7"]
        self.CREDIBLE_UNITS = ["B", "M", "%"]

        # Attack templates — crafted to look realistic
        self.FN_PATTERNS = [
            "{entity} reports {metric} of ${value}{unit}, beating estimates",
            "{entity} announces strategic partnership with {partner}",
            "{entity} secures regulatory approval for {product} in {region}",
            "Sources say {entity} in talks to acquire {target} for ${value}B",
            "{entity} CFO confirms guidance raise for FY{year}",
            "{entity} receives upgraded rating from analyst as {reason}",
            "Breaking: {entity} board approves {action} to unlock shareholder value",
            "{entity} patent filing reveals breakthrough in {technology}",
        ]

        self.BOT_SOCIAL_TEMPLATES = [
            "Just read that {entity} is about to {action} — this is huge if true 🚀",
            "Can anyone confirm the {entity} news? Seeing multiple sources now.",
            "My source at {entity} confirms the {claim} is accurate ✅",
            "Why is nobody talking about {entity} {claim}? This is game-changing.",
            "Heard from a reliable source: {entity} {action} is happening next week.",
        ]

    def generate_hoax_payload(self, round_num=None, target_pattern=None) -> Dict:
        """Generate a single adversarial hoax headline + social stream.

        Args:
            round_num: Generation round (higher = more sophisticated)
            target_pattern: Optional specific FN pattern to exploit

        Returns:
            dict with headline, entity, social_posts, metadata
        """
        self.round = round_num or (self.round + 1)
        rng = self.rng
        entity = self.py_random.choice(self.REAL_ENTITIES)

        # Generate headline
        if target_pattern:
            template = target_pattern
        else:
            template = self.py_random.choice(self.FN_PATTERNS)

        partner = self.py_random.choice([e for e in self.REAL_ENTITIES if e != entity])
        target = partner

        headline = template.format(
            entity=entity,
            partner=partner,
            target=target,
            metric=self.py_random.choice(["revenue", "EPS", "operating margin", "FCF"]),
            value=self.py_random.choice(self.CREDIBLE_VALUES),
            unit=self.py_random.choice(self.CREDIBLE_UNITS),
            product="AI-powered analytics platform",
            region="European Union",
            year=2026,
            reason="strong Q4 performance",
            action="stock buyback",
            technology="quantum-resistant encryption",
            claim="major partnership",
        )

        # Generate social media posts
        social_posts = []
        for i in range(np.random.randint(3, 7)):
            post_template = self.py_random.choice(self.BOT_SOCIAL_TEMPLATES)
            post_text = post_template.format(
                entity=entity,
                action="about to announce a major acquisition",
                claim="breakthrough technology",
            )
            timestamp = np.random.uniform(1.0, 30.0)
            social_posts.append({
                "text": post_text,
                "timestamp_seconds": round(timestamp, 2),
                "author": f"@AdversarialBot_{i:03d}",
                "post_type": "bot_amplify",
            })

        return {
            "headline": headline,
            "entity": entity,
            "is_fake": True,
            "round": self.round,
            "social_posts": social_posts,
            "sophistication": min(1.0, self.round * 0.15),  # increases with rounds
        }

    def generate_batch(self, n=10, round_num=None) -> pd.DataFrame:
        """Generate a batch of adversarial headlines for testing."""
        payloads = [self.generate_hoax_payload(round_num=round_num) for _ in range(n)]
        rows = []
        for i, p in enumerate(payloads):
            rows.append({
                "event_id": f"RED-{self.round:02d}-{i:04d}",
                "T0_headline": p["headline"],
                "T2_human_verdict": 1,  # all generated payloads are FAKE
                "entity": p["entity"],
                "sophistication": p["sophistication"],
                "red_team_round": self.round,
            })
        return pd.DataFrame(rows)

    def learn_from_failures(self, pipeline_results):
        """Analyze pipeline results to identify FN patterns for next round.

        Args:
            pipeline_results: list of result dicts from MFTPipeline.process_events()

        Updates self.failure_history with patterns that caused false negatives.
        """
        fns = [r for r in pipeline_results if r.get("outcome") == "false_negative"]
        for fn in fns:
            headline = fn.get("headline", "")
            entity = fn.get("entity", "")
            confidence = fn.get("llm_confidence", 0)
            self.failure_history.append({
                "headline": headline,
                "entity": entity,
                "confidence": confidence,
            })

        n_fn = len(fns)
        print(f"[RedTeam] Round {self.round}: {n_fn} FNs identified, "
              f"{len(self.failure_history)} total in history")
        return n_fn > 0


if __name__ == "__main__":
    print("Red-Team Co-Evolution Demo")
    print("=" * 60)

    attacker = RedTeamGenerator(seed=42)

    # Round 1: Generate baseline adversarial payloads
    print("\nRound 1: Generating baseline adversarial payloads...")
    batch = attacker.generate_batch(n=5, round_num=1)
    for _, row in batch.iterrows():
        print(f"  {row['event_id']}: '{row['T0_headline'][:60]}...' [R{row['red_team_round']}]")

    # Round 2: More sophisticated (simulate learning)
    print("\nRound 2: Targeted adversarial generation...")
    for i in range(3):
        payload = attacker.generate_hoax_payload(round_num=2)
        print(f"  R2-{i}: '{payload['headline'][:60]}...' ({payload['entity']})")
        print(f"    Social: {len(payload['social_posts'])} bot posts")

    print(f"\nTotal failure patterns learned: {len(attacker.failure_history)}")
