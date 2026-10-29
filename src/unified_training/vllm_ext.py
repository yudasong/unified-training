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

"""vLLM worker extension for in-process weight reload from an HF safetensors dir.

The generator's weight sync is disk-mediated: the trainer publishes
policies/vNNNNNNN/ and the generator loads it into the live engine with
``update_weights_from_disk``. vLLM injects these methods into the GPU worker via
the ``worker_extension_cls`` engine kwarg, so inside the methods ``self`` IS the
vLLM worker: ``self.model_runner.model`` is the live model whose per-parameter
weight loaders handle any tensor-parallel sharding. Adapted from NeMo-RL's
refit worker extension (Apache-2.0).

The caller MUST follow a reload with the engine-level cache invalidation
(``llm.llm_engine.reset_prefix_cache()`` + gc + empty_cache) so all KV is
recomputed under the new parameters; the generator does this between batches,
when no requests are in flight.
"""

from __future__ import annotations

import gc
import os


class UnifiedWorkerExtension:
    """Disk reload of a policy snapshot (call via ``collective_rpc`` by name)."""

    def update_weights_from_disk(self, hf_dir: str, read_mode: str = "mmap") -> bool:
        """Load all *.safetensors under ``hf_dir`` into the live model.

        Shard-by-shard to bound CPU memory; finishes with vLLM's
        ``process_weights_after_loading``.
        """
        import torch
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        from unified_training.policy_weights import load_policy_shard

        shards = sorted(
            os.path.join(hf_dir, f) for f in os.listdir(hf_dir) if f.endswith(".safetensors")
        )
        if not shards:
            raise FileNotFoundError(f"no *.safetensors under {hf_dir}")
        for shard in shards:
            state = load_policy_shard(shard, read_mode)  # CPU tensors; vLLM loaders move/shard them
            self.model_runner.model.load_weights(weights=list(state.items()))
            del state
        process_weights_after_loading(self.model_runner.model, self.model_config, self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return True
