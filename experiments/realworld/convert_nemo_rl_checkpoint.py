#!/usr/bin/env python3
"""Convert one NeMo-RL policy checkpoint (OUT/checkpoints/step_N) to an HF directory for evaluation or as an init.

  python convert_nemo_rl_checkpoint.py --source-checkpoint OUT/checkpoints/step_4000 --optimizer-step 4000 --output DIR
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def expected_receipt(source: Path, optimizer_step: int) -> dict:
    return {
        "schema_version": 1,
        "source_checkpoint": str(source),
        "optimizer_step": optimizer_step,
        "source_config_sha256": file_sha256(source / "config.yaml"),
        "source_dcp_metadata_sha256": file_sha256(
            source / "policy" / "weights" / ".metadata"
        ),
    }


def validate_existing(output: Path, expected: dict) -> None:
    receipt_path = output / "conversion.json"
    milestone_path = output / "milestone.json"
    weights = output / "model.safetensors"
    required = (
        output / "config.json",
        output / "generation_config.json",
        output / "tokenizer.json",
        output / "tokenizer_config.json",
        weights,
        receipt_path,
        milestone_path,
        output / "CONVERSION_DONE",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete existing converted checkpoint {output}: {missing}")
    receipt = json.loads(receipt_path.read_text())
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"converted checkpoint provenance conflict {output}/{key}: "
                f"expected {value!r}, got {receipt.get(key)!r}"
            )
    if receipt.get("model_safetensors_sha256") != file_sha256(weights):
        raise ValueError(f"converted checkpoint weight digest changed: {weights}")
    milestone = json.loads(milestone_path.read_text())
    if milestone.get("actual_optimizer_step") != expected["optimizer_step"]:
        raise ValueError(f"converted checkpoint milestone conflict: {milestone_path}")


def convert(source: Path, optimizer_step: int, output: Path) -> None:
    source = source.resolve(strict=True)
    output = output.resolve(strict=False)
    if not source.is_dir():
        raise ValueError(f"source checkpoint is not a directory: {source}")
    config_path = source / "config.yaml"
    dcp_path = source / "policy" / "weights"
    tokenizer_path = source / "policy" / "tokenizer"
    for required in (
        config_path,
        dcp_path / ".metadata",
        tokenizer_path / "tokenizer.json",
    ):
        if not required.exists():
            raise ValueError(f"source checkpoint is incomplete: {required}")
    expected = expected_receipt(source, optimizer_step)
    if output.exists():
        validate_existing(output, expected)
        print(f"already converted and validated: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.converting-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"conversion staging directory already exists: {temporary}")
    temporary.mkdir()
    try:
        torch_path = temporary / "pytorch_model.bin"
        dcp_to_torch_save(str(dcp_path), str(torch_path))
        state = torch.load(torch_path, map_location="cpu", weights_only=True, mmap=True)
        if set(state) == {"model"} and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict) or not state:
            raise ValueError("DCP conversion did not produce a non-empty state dict")
        invalid = [
            key
            for key, value in state.items()
            if not isinstance(key, str) or not torch.is_tensor(value)
        ]
        if invalid:
            raise ValueError(f"DCP conversion produced invalid state entries: {invalid[:10]}")
        tensors = {key: value.contiguous() for key, value in state.items()}
        weights_path = temporary / "model.safetensors"
        save_file(tensors, weights_path, metadata={"format": "pt"})
        tensor_count = len(tensors)
        del tensors, state
        gc.collect()
        torch_path.unlink()

        config_payload = yaml.safe_load(config_path.read_text())
        model_name = Path(config_payload["policy"]["model_name"]).resolve(strict=True)
        hf_overrides = config_payload["policy"].get("hf_overrides", {}) or {}
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            **hf_overrides,
        )
        config.save_pretrained(temporary)
        generation_config = model_name / "generation_config.json"
        if not generation_config.is_file():
            raise ValueError(f"source model lacks generation config: {generation_config}")
        shutil.copy2(generation_config, temporary / generation_config.name)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer.save_pretrained(temporary)
        chat_template = tokenizer_path / "chat_template.jinja"
        if chat_template.is_file() and not (temporary / chat_template.name).is_file():
            shutil.copy2(chat_template, temporary / chat_template.name)

        atomic_json(
            temporary / "milestone.json",
            {
                "schema_version": 1,
                "actual_optimizer_step": optimizer_step,
                "source_checkpoint": str(source),
            },
        )

        # A real HF load catches key/shape/config mismatches before atomic commit.
        model = AutoModelForCausalLM.from_pretrained(
            temporary,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        parameter_count = model.num_parameters()
        del model
        gc.collect()

        receipt = {
            **expected,
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "model_safetensors_bytes": weights_path.stat().st_size,
            "model_safetensors_sha256": file_sha256(weights_path),
            "source_model": str(model_name),
        }
        atomic_json(temporary / "conversion.json", receipt)
        (temporary / "CONVERSION_DONE").write_text("complete\n")
        os.replace(temporary, output)
        validate_existing(output, expected)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--optimizer-step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.optimizer_step < 0:
        parser.error("--optimizer-step must be non-negative")
    convert(args.source_checkpoint, args.optimizer_step, args.output)


if __name__ == "__main__":
    main()
