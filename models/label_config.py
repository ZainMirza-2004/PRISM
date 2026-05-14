"""Label configuration for social bias classification."""

LABELS = ["gender_bias", "nationality_bias", "profession_bias", "neutral"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
NEUTRAL_LABEL = "neutral"
