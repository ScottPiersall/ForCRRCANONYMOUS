#!/usr/bin/env python
"""Pin and validate the 7B and 32B checkpoints before any S3 row is scored.

Runs after the preregistration is frozen and before inference. For each new
checkpoint it resolves the exact Hugging Face commit, records provenance, and
establishes protocol continuity with the frozen Qwen2.5-14B run.

What this does with the S3 panel is deliberately limited to *rendering*: the
2,916 prompts are passed through the checkpoint's chat template and tokenizer to
confirm that the rendered text is byte-identical to what the frozen 14B run
recorded and that the A/B continuation is the same single token everywhere. No
forward pass is run on any S3 prompt and no S3 outcome exists when this ends.
The only prompts scored are the permanently designated non-study development
prompts in ``qwen_scale_config.DEVELOPMENT_PROMPTS``.

It also reloads the frozen 14B checkpoint and re-scores its recorded smoke
prompt. Reproducing those two numbers shows the numerical stack is the one
Day 3 ran on. That is not a rerun of 14B on S3.

Writes ``preflight.json`` and ``model_revisions.json``. A checkpoint whose
record is not PASS cannot be run: the inference script refuses it.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc import prompts, provenance, scoring  # noqa: E402
from formalcrrc import qwen_scale_config as qc  # noqa: E402
from formalcrrc.config import LABEL_MET, LABEL_NOT_MET  # noqa: E402


def _release(model: Any) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def _peak_gib() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def _load(model_id: str, revision: str, token: str | None, cache_dir: str | None):
    """The frozen Day-3 loader, reused so the load path cannot drift."""
    from run_day3_inference import load_model

    return load_model(model_id, revision, token, cache_dir)


def render_continuity(tokenizer: Any, root: Path) -> dict[str, Any]:
    """Render every S3 prompt and compare against the frozen 14B record.

    Tokenizer only. Nothing is scored.
    """
    manifest = pd.read_parquet(root / d3c.PROMPT_MANIFEST_PATH).sort_values(
        "order_index"
    )
    frozen = pd.read_parquet(root / qc.REFERENCE_RAW_SCORES_PATH).set_index(
        "prompt_id"
    )["rendered_prompt_sha256"]

    reference_prompt = scoring.render_chat_prompt(
        tokenizer, prompts.label_reference_message()
    )
    reference_tok = scoring.resolve_label_tokenization(tokenizer, reference_prompt)

    render_mismatches: list[str] = []
    token_mismatches: list[str] = []
    for entry in manifest.itertuples(index=False):
        rendered = scoring.render_chat_prompt(tokenizer, entry.user_message)
        if scoring.sha256_text(rendered) != frozen[entry.prompt_id]:
            render_mismatches.append(entry.prompt_id)
        try:
            met = scoring.label_continuation_ids(tokenizer, rendered, LABEL_MET)
            not_met = scoring.label_continuation_ids(tokenizer, rendered, LABEL_NOT_MET)
        except ValueError:
            token_mismatches.append(entry.prompt_id)
            continue
        if met != reference_tok.met_token_ids or not_met != reference_tok.not_met_token_ids:
            token_mismatches.append(entry.prompt_id)

    return {
        "n_prompts_rendered": int(len(manifest)),
        "n_render_mismatches": len(render_mismatches),
        "render_mismatch_examples": render_mismatches[:5],
        "n_label_tokenization_mismatches": len(token_mismatches),
        "label_tokenization_mismatch_examples": token_mismatches[:5],
        "label_tokenization": reference_tok.to_dict(),
        "decoded_met_token": tokenizer.decode(list(reference_tok.met_token_ids)),
        "decoded_not_met_token": tokenizer.decode(list(reference_tok.not_met_token_ids)),
    }


def check_reference(root: Path, token: str | None, cache_dir: str | None) -> dict[str, Any]:
    """Reload the frozen 14B and reproduce its recorded smoke-prompt scores."""
    model_id = qc.MODEL_IDS[qc.REFERENCE_TAG]
    record: dict[str, Any] = {"model_id": model_id, "revision": qc.REFERENCE_REVISION}
    tokenizer, model = _load(model_id, qc.REFERENCE_REVISION, token, cache_dir)
    rendered = scoring.render_chat_prompt(tokenizer, qc.SMOKE_PROMPT)
    tokenization = scoring.resolve_label_tokenization(
        tokenizer,
        scoring.render_chat_prompt(tokenizer, prompts.label_reference_message()),
    )
    result = scoring.score_prompt(model, tokenizer, rendered, tokenization)
    record.update(
        {
            "chat_template_sha256": provenance.sha256_text(tokenizer.chat_template or ""),
            "smoke_raw_score_met": result["raw_score_met"],
            "smoke_raw_score_not_met": result["raw_score_not_met"],
            "expected_raw_score_met": qc.REFERENCE_SMOKE_SCORE_MET,
            "expected_raw_score_not_met": qc.REFERENCE_SMOKE_SCORE_NOT_MET,
            "is_study_prompt": False,
        }
    )
    record["checks"] = {
        "chat_template_matches_frozen": record["chat_template_sha256"]
        == qc.REFERENCE_CHAT_TEMPLATE_SHA256,
        "smoke_scores_reproduce": result["raw_score_met"]
        == qc.REFERENCE_SMOKE_SCORE_MET
        and result["raw_score_not_met"] == qc.REFERENCE_SMOKE_SCORE_NOT_MET,
    }
    record["status"] = "PASS" if all(record["checks"].values()) else "FAIL"
    _release(model)
    return record


def check_new(
    tag: str, root: Path, token: str | None, cache_dir: str | None
) -> dict[str, Any]:
    """Resolve, record and validate one new checkpoint."""
    import torch
    from huggingface_hub import HfApi, snapshot_download

    model_id = qc.MODEL_IDS[tag]
    record: dict[str, Any] = {
        "tag": tag,
        "model_id": model_id,
        "short_name": qc.MODEL_SHORT_NAMES[tag],
        "checked_at": provenance.utc_now(),
        "checks": {},
        "problems": [],
    }

    def fail(message: str) -> None:
        record["problems"].append(message)
        print(f"PROBLEM [{tag}]: {message}", flush=True)

    info = HfApi().model_info(model_id, revision="main", token=token)
    revision = info.sha
    card = info.card_data.to_dict() if info.card_data is not None else {}
    record.update(
        {
            "revision": revision,
            "tokenizer_revision": revision,
            "revision_source": "Hugging Face `main` resolved at pre-flight",
            "license": card.get("license"),
            "license_link": card.get("license_link"),
            "base_model": card.get("base_model"),
            "hub_last_modified": str(getattr(info, "last_modified", None)),
        }
    )
    record["checks"]["revision_is_full_commit_hash"] = (
        isinstance(revision, str) and len(revision) == 40
    )
    print(f"[{tag}] {model_id} @ {revision}", flush=True)

    t0 = time.time()
    local = Path(
        snapshot_download(
            model_id, revision=revision, token=token, cache_dir=cache_dir, max_workers=4
        )
    )
    record["snapshot_seconds"] = round(time.time() - t0, 1)

    config = json.loads((local / "config.json").read_text(encoding="utf-8"))
    record["config"] = config
    record["config_sha256"] = provenance.sha256_file(local / "config.json")
    for name in ("tokenizer_config.json", "tokenizer.json", "generation_config.json"):
        if (local / name).is_file():
            record[f"{name.replace('.', '_')}_sha256"] = provenance.sha256_file(
                local / name
            )
    if (local / "LICENSE").is_file():
        record["license_file_sha256"] = provenance.sha256_file(local / "LICENSE")
    if (local / "tokenizer_config.json").is_file():
        tok_cfg = json.loads((local / "tokenizer_config.json").read_text(encoding="utf-8"))
        record["tokenizer_config"] = {
            k: v for k, v in tok_cfg.items() if k != "added_tokens_decoder"
        }
    record["weight_files"] = sorted(
        p.name for p in local.glob("*") if p.suffix in (".safetensors", ".bin")
    )
    record["checks"]["no_quantization_config"] = "quantization_config" not in config
    banned = [
        p.name
        for p in local.glob("*")
        for marker in qc.FORBIDDEN_QUANTIZATION
        if marker in p.name.lower()
    ]
    record["checks"]["no_quantized_files"] = not banned

    # --- tokenizer and rendering continuity: tokenizer only, nothing scored ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token, cache_dir=cache_dir
    )
    record["chat_template_sha256"] = provenance.sha256_text(tokenizer.chat_template or "")
    record["chat_template_identical_to_14b"] = (
        record["chat_template_sha256"] == qc.REFERENCE_CHAT_TEMPLATE_SHA256
    )
    continuity = render_continuity(tokenizer, root)
    record["render_continuity"] = continuity
    record["label_tokenization"] = continuity["label_tokenization"]
    tokenization = continuity["label_tokenization"]
    record["checks"].update(
        {
            "labels_are_single_token": bool(tokenization["single_token_equivalence"]),
            "scoring_method_is_single_token_next_logit": tokenization["scoring_method"]
            == qc.REQUIRED_SCORING_METHOD,
            "label_tokens_decode_to_A_and_B": continuity["decoded_met_token"] == LABEL_MET
            and continuity["decoded_not_met_token"] == LABEL_NOT_MET,
            "label_tokenization_stable_on_all_s3_prompts": continuity[
                "n_label_tokenization_mismatches"
            ]
            == 0,
            "all_s3_prompts_render_byte_identical_to_14b": continuity[
                "n_prompts_rendered"
            ]
            == qc.N_ROWS
            and continuity["n_render_mismatches"] == 0,
        }
    )
    if not all(record["checks"].values()):
        fail("tokenizer / rendering continuity with the frozen 14B protocol failed")
        record["status"] = "FAIL"
        return record

    # --- unquantized bf16 load on one GPU, development prompts only ---
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        _, model = _load(model_id, revision, token, cache_dir)
    except Exception as error:  # noqa: BLE001 - reported, never worked around
        fail(f"bf16 load failed: {type(error).__name__}: {str(error)[:300]}")
        record["checks"]["loads_in_bf16_on_one_gpu"] = False
        record["status"] = "FAIL"
        return record
    record["load_seconds"] = round(time.time() - t0, 1)
    record["checks"]["loads_in_bf16_on_one_gpu"] = True
    record["dtype"] = str(next(model.parameters()).dtype)
    record["checks"]["weights_are_bfloat16"] = record["dtype"] == "torch.bfloat16"
    record["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    record["all_parameters_on_gpu"] = all(p.is_cuda for p in model.parameters())
    record["checks"]["no_offloading"] = record["all_parameters_on_gpu"]

    resolved = scoring.resolve_label_tokenization(
        tokenizer,
        scoring.render_chat_prompt(tokenizer, prompts.label_reference_message()),
    )
    development: list[dict[str, Any]] = []
    try:
        for text in qc.DEVELOPMENT_PROMPTS:
            rendered = scoring.render_chat_prompt(tokenizer, text)
            dev_tok = scoring.resolve_label_tokenization(tokenizer, rendered)
            result = scoring.score_prompt(model, tokenizer, rendered, resolved)
            development.append(
                {
                    "prompt": text,
                    "is_study_prompt": False,
                    "label_tokenization_matches_reference": dev_tok.to_dict()
                    == resolved.to_dict(),
                    **result,
                    "margin": result["raw_score_met"] - result["raw_score_not_met"],
                }
            )
    except Exception as error:  # noqa: BLE001
        fail(f"development scoring failed: {type(error).__name__}: {str(error)[:300]}")
    record["development_prompts"] = development
    record["checks"]["development_prompts_scored"] = len(development) == len(
        qc.DEVELOPMENT_PROMPTS
    )
    record["checks"]["development_label_tokenization_stable"] = all(
        d["label_tokenization_matches_reference"] for d in development
    )
    record["peak_gpu_memory_gib"] = _peak_gib()
    record["transformers_version"] = __import__("transformers").__version__
    record["torch_version"] = torch.__version__
    _release(model)

    record["status"] = (
        "PASS" if all(record["checks"].values()) and not record["problems"] else "FAIL"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--hf-token-file", default=os.environ.get("HF_TOKEN_FILE"))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    root = Path(args.root)

    prereg, sha = root / qc.PREREG_JSON_PATH, root / qc.PREREG_SHA_PATH
    if not (prereg.is_file() and sha.is_file()) or not provenance.verify_checksum_file(
        prereg, sha
    ):
        print("PREFLIGHT FAIL: the preregistration is not frozen and verified", flush=True)
        return 2
    existing = [t for t in qc.NEW_TAGS if (root / qc.results_path(t)).exists()]
    if existing:
        print(f"PREFLIGHT FAIL: inference already exists for {existing}", flush=True)
        return 2

    token = None
    if args.hf_token_file and Path(args.hf_token_file).is_file():
        token = Path(args.hf_token_file).read_text(encoding="utf-8").strip() or None

    environment = provenance.environment_snapshot(include_gpu=True)
    packages = environment.get("packages", {})
    stack_checks = {
        f"{name}_matches_frozen_14b_run": packages.get(name) == version
        for name, version in qc.REFERENCE_STACK.items()
    }
    devices = environment.get("gpu", {}).get("devices", [])
    stack_checks["single_h100_80gb"] = (
        len(devices) == 1
        and "H100" in devices[0].get("name", "")
        and devices[0].get("total_memory_bytes", 0) >= 79 * 1024**3
    )

    report: dict[str, Any] = {
        "study": qc.STUDY_NAME,
        "purpose": qc.PREFLIGHT_SCOPE,
        "scores_any_s3_prompt": False,
        "preregistration_sha256": sha.read_text(encoding="utf-8").split()[0],
        "started_at": provenance.utc_now(),
        "environment": environment,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "stack_checks": stack_checks,
    }

    print("reference: reloading frozen 14B for the smoke prompt ...", flush=True)
    report["reference_14b"] = check_reference(root, token, args.cache_dir)
    print(f"reference status {report['reference_14b']['status']}", flush=True)

    shared_ok = all(stack_checks.values()) and report["reference_14b"]["status"] == "PASS"
    report["models"] = {}
    for tag in qc.RUN_ORDER:
        record = check_new(tag, root, token, args.cache_dir)
        if not shared_ok:
            record["problems"].append(
                "stack continuity with the frozen 14B run was not established"
            )
            record["status"] = "FAIL"
        report["models"][tag] = record
        print(f"[{tag}] status {record['status']}", flush=True)
        for name, ok in record["checks"].items():
            print(f"    {'PASS' if ok else 'FAIL'}  {name}", flush=True)

    report["finished_at"] = provenance.utc_now()
    report["status"] = (
        "PASS"
        if shared_ok and all(r["status"] == "PASS" for r in report["models"].values())
        else "FAIL"
    )
    provenance.write_json(root / qc.PREFLIGHT_PATH, report)

    reference_run = json.loads(
        (root / qc.REFERENCE_RUN_RECORD_PATH).read_text(encoding="utf-8")
    )
    revisions = {
        "study": qc.STUDY_NAME,
        "pinned_at": report["finished_at"],
        "preregistration_sha256": report["preregistration_sha256"],
        "revision_rule": qc.REVISION_RULE,
        "preflight_slurm": report["slurm"],
        "models": {
            qc.REFERENCE_TAG: {
                "tag": qc.REFERENCE_TAG,
                "model_id": qc.MODEL_IDS[qc.REFERENCE_TAG],
                "role": "existing frozen S3 result; not rerun",
                "revision": qc.REFERENCE_REVISION,
                "tokenizer_revision": qc.REFERENCE_TOKENIZER_REVISION,
                "chat_template_sha256": qc.REFERENCE_CHAT_TEMPLATE_SHA256,
                "label_tokenization": reference_run["label_tokenization"],
                "source": qc.REFERENCE_RUN_RECORD_PATH,
            },
            **{tag: dict(record) for tag, record in report["models"].items()},
        },
    }
    provenance.write_json(root / qc.MODEL_REVISIONS_PATH, revisions)
    provenance.write_checksum_file(
        root / qc.MODEL_REVISIONS_PATH, root / qc.MODEL_REVISIONS_SHA_PATH
    )

    print(f"\nwrote {qc.PREFLIGHT_PATH} and {qc.MODEL_REVISIONS_PATH}", flush=True)
    for name, ok in stack_checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    print(f"PREFLIGHT STATUS: {report['status']}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
