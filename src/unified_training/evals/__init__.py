"""Benchmark harness for the real-world (0.76B) experiments, shared with the controlled setting.

Protocol (Appendix D.1): MATH-500 / GSM8K pass@1 from 16 samples and pass@32/64/256 from 256 samples;
HumanEval+ / MBPP+ (EvalPlus) pass@1 from 16 samples and pass@32 from 32 samples; sampling T=1, top-k 50,
top-p 0.95, context 2048, 1024 (math) / 768 (code) new tokens; SciQ / ARC-Challenge greedy with the choices
in the prompt (8 new tokens, strict leading label); MMLU 0-shot answer log-likelihoods; held-out midtraining
loss averaged with equal weight over the 24 corpus components. CLI: ``python -m unified_training.evals -h``.
"""
