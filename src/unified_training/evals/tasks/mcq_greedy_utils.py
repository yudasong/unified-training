"""lm-eval `!function` hooks for sciq_greedy / arc_challenge_greedy (implementation in unified_training.evals)."""
from unified_training.evals.mcq import (  # noqa: F401
    doc_to_greedy_mcq,
    doc_to_mcq_target,
    process_arc_docs,
    process_greedy_mcq_results,
    process_sciq_docs,
)
