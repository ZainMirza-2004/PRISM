"""Main batch pipeline: hybrid Cardiff RoBERTa (API) + local bias type head + light cues."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from data.loaders import load_dataset
from models.hybrid_pipeline import HybridBiasPipeline
from utils.io import save_results

LOGGER = logging.getLogger(__name__)
DEFAULT_TYPE_DIR = "models/distilbert_B_balanced"


def analyze_posts(
    dataset_path: str,
    model_dir: str = DEFAULT_TYPE_DIR,
    output_path: Optional[str] = None,
    confidence_threshold: float = 0.0,
    batch_size: int = 8,
    hf_token: Optional[str] = None,
    hate_model_id: Optional[str] = None,
    *,
    meta_classifier_path: Optional[str] = None,
    auxiliary_type_model_dir: Optional[str] = None,
    legacy_fusion: bool = False,
    calibrated: bool = False,
    use_thresholds: bool = False,
    thresholds_path: Optional[str] = None,
) -> List[Dict]:
    """Run hybrid bias analysis. *confidence_threshold* is kept for CLI compatibility; filtering is in the decision layer.

    **HF API**: set *hf_token* or `HF_API_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` for higher rate limits.
    """
    _ = confidence_threshold
    records = load_dataset(dataset_path)
    token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    meta_path = "" if legacy_fusion else meta_classifier_path
    pipeline = HybridBiasPipeline(
        model_dir,
        hf_token=token,
        hate_model_id=hate_model_id,
        meta_classifier_path=meta_path,
        auxiliary_type_model_dir=auxiliary_type_model_dir,
        calibrated=calibrated,
        use_thresholds=use_thresholds,
        thresholds_path=thresholds_path,
    )
    texts = [str(r.get("text", "")) for r in records]
    mode = "local-only DistilBERT + rules" if pipeline.hate_disabled else "API + local type head"
    LOGGER.info("Running hybrid model on %s posts (%s).", len(texts), mode)
    out = pipeline.predict_batch(records, batch_size=batch_size)
    for row in out:
        print(
            {
                "post_id": row.get("post_id"),
                "bias_type": row.get("bias_type"),
                "bias_detected": row.get("bias_detected"),
                "confidence": row.get("confidence"),
            }
        )
    if output_path:
        output_file = save_results(out, output_path)
        LOGGER.info("Saved %s results to %s", len(out), output_file)
    return out
