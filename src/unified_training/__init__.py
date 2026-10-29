# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified training: static data and RL generations under one weighted likelihood loss.

Entry points (configured by UNIFIED_* environment variables, docs/configuration.md):
  * ``unified_training.gen_service`` -- vLLM generator: samples G completions per prompt,
    scores them, weights them (MaxRL 1/p_hat or REINFORCE) and appends rounds to the buffer.
  * ``unified_training.hf_trainer`` -- torchrun trainer: mixes the RL rounds with static
    (anchor / pretraining) rows and optimizes Eq. (2); publishes weights every k updates.
  * ``scripts/train_unified.sh`` -- runs both on one node.

Subpackages: ``tools`` (data builders writing the round format), ``pretrain`` (124M pretraining
data and staged SFT), ``evals`` (benchmark harness).

Module layering: ``episodes`` (shard format) and ``rollout_common`` are numpy-only;
``dataset`` adds torch; ``hf_trainer`` adds transformers; ``gen_service`` adds vLLM.
"""
