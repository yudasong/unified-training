"""lm-eval `!function` hooks for math500_16 / gsm8k_16 (implementation in unified_training.evals)."""
from unified_training.evals.math_answers import (  # noqa: F401
    doc_to_gsm8k,
    doc_to_math,
    gsm8k_gold_answer,
    process_gsm8k_results,
    process_math_results,
)
