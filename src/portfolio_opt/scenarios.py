"""Scenario definitions for stochastic modeling."""

SCENARIOS = {
    "optimistic": {
        "prob": 0.2,
        "resale_multiplier": 1.05,
        "rehab_multiplier": 0.90,
    },
    "base": {
        "prob": 0.5,
        "resale_multiplier": 1.00,
        "rehab_multiplier": 1.00,
    },
    "pessimistic": {
        "prob": 0.3,
        "resale_multiplier": 0.90,
        "rehab_multiplier": 1.15,
    },
}
