"""nl_tower: the reasoning puzzles of the controlled experiments (paper Section 5.1, Appendices B-C).

Modules (standard library + numpy; nothing here imports torch, except vLLM inside ``evaluate``):
  * ``generator``, ``oracle``, ``specs`` - puzzle generator, deterministic solver, puzzle distributions;
  * ``traces``, ``pools``, ``mixes``, ``demos`` - midtraining corpus, test sets, RL prompt mixes, demo rows;
  * ``prepare_data`` (CLI) and ``manifest`` - build the data from seeds and check it against ``artifacts.json``;
  * ``answers``, ``passk``, ``evaluate`` (CLI) - scoring and the pass@k evaluator;
  * ``results``, ``compute`` - two-seed averaging and the compute estimates of Appendix A.2.
"""
