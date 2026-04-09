"""Project-wide constants."""

STAGE_ORDER = [
    "S1_literature",
    "S2_plan",
    "S3_experiment",
    "S4_writing",
    "S5_review",
]

STAGE_LABELS = {
    "S1_literature": "literature",
    "S2_plan": "plan",
    "S3_experiment": "experiment",
    "S4_writing": "paper",
    "S5_review": "review",
}

REQUIRED_STAGE_DIRS = [
    "references",
    "literature",
    "plan",
    "experiment",
    "paper",
    "review",
    "meta",
]
