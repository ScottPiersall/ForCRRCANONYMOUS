#!/usr/bin/env python
"""Score the frozen S3 panel with one new Qwen2.5 rung (7B or 32B).

Raw scores only. No metric is computed and nothing scientific is printed: the
console carries progress, failures, timing and memory, so that the 7B run cannot
inform the 32B run or the frozen analysis.

The scoring path is the frozen immediate-readout implementation, reused without
modification: the s-3 model loader, ``scoring.render_chat_prompt`` and
``scoring.score_prompt``, one forward pass per rubric condition, batch size 1,
in the frozen s-3 inference order.

Fails closed unless the preregistration is frozen and verified, the S3 dataset
and prompt manifest match their manifest, the protocol-defining source files are
byte-identical to the ones the preregistration hashed, the checkpoint was pinned
and passed the pre-flight, and every rendered prompt is byte-identical to the
one the frozen 14B run recorded.

A condition that already has a stored margin is never scored again. The script
refuses to start when the model's final result file exists, and when resuming
after an infrastructure failure it skips every prompt_id already in the partial
file. Each invocation appends a receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from formalcrrc import s3_config as d3c  # noqa: E402
from formalcrrc import prompts, provenance, scoring  # noqa: E402
from formalcrrc import qwen_scale as qs  # noqa: E402
from formalcrrc import qwen_scale_config as qc  # noqa: E402
from formalcrrc.config import INFERENCE_BATCH_SIZE, MODEL_DTYPE, N_ROWS  # noqa: E402

RAW_SCORE_COLUMNS = [
    "prompt_id",
    "artifact_id",
    "family",
    "latent_level",
    "strictness_index",
    "order_index",
    "partition",
    "model_id",
    "revision",
    "raw_score_met",
    "raw_score_not_met",
    "margin",
    "p_met",
    "scoring_method",
    "n_prompt_tokens",
    "rendered_prompt_sha256",
]

#: Files that define what a score means. Must match the frozen preregistration.
PROTOCOL_SOURCE_PATHS = (
    "src/formalcrrc/config.py",
    "src/formalcrrc/prompts.py",
    "src/formalcrrc/scoring.py",
    "src/formalcrrc/qwen_scale_config.py",
    "scripts/run_qwen_scale_inference.py",
    "scripts/run_s3_inference.py",
    "src/formalcrrc/qwen_scale.py",
)


def gate(root: Path, tag: str) -> dict:
    """Every precondition, checked before a single prompt is scored."""
    problems: list[str] = []

    prereg_path, sha_path = root / qc.PREREG_JSON_PATH, root / qc.PREREG_SHA_PATH
    prereg: dict = {}
    if not (prereg_path.is_file() and sha_path.is_file()):
        problems.append("the scale-ablation preregistration is not frozen")
    elif not provenance.verify_checksum_file(prereg_path, sha_path):
        problems.append("the preregistration checksum does not match")
    else:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        try:
            manifest_in_force = qs.effective_code_manifest(root)
        except ValueError as error:
            problems.append(str(error))
            manifest_in_force = {"files": {}, "source": "unavailable"}
        frozen_sources = manifest_in_force["files"]
        for relative in PROTOCOL_SOURCE_PATHS:
            if relative not in frozen_sources:
                problems.append(f"{relative} is not pinned by {manifest_in_force['source']}")
            elif provenance.sha256_file(root / relative) != frozen_sources[relative]:
                problems.append(
                    f"{relative} differs from the {manifest_in_force['source']} manifest"
                )

    manifest = json.loads((root / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8"))
    if provenance.sha256_file(root / d3c.DATASET_PATH) != manifest["hashes"][
        "dataset_file_sha256"
    ]:
        problems.append("S3 dataset hash does not match its manifest")
    if provenance.sha256_file(root / d3c.PROMPT_MANIFEST_PATH) != manifest["hashes"][
        "prompt_manifest_file_sha256"
    ]:
        problems.append("S3 prompt manifest hash does not match its manifest")
    if manifest["hashes"]["prompt_template_sha256"] != prompts.prompt_template_sha256():
        problems.append("prompt template hash has changed")
    if provenance.sha256_file(root / qc.REFERENCE_RAW_SCORES_PATH) != (
        qc.REFERENCE_RAW_SCORES_SHA256
    ):
        problems.append("the frozen 14B raw scores have changed")

    record: dict = {}
    revisions_path = root / qc.MODEL_REVISIONS_PATH
    revisions_sha_path = root / qc.MODEL_REVISIONS_SHA_PATH
    if not revisions_path.is_file():
        problems.append("model_revisions.json missing; run preflight_qwen_scale.py")
    elif not revisions_sha_path.is_file():
        problems.append("model_revisions.sha256 missing; the pinned revisions are unverified")
    elif not provenance.verify_checksum_file(revisions_path, revisions_sha_path):
        problems.append("model_revisions.json does not match its checksum")
    else:
        document = json.loads(revisions_path.read_text(encoding="utf-8"))
        record = document.get("models", {}).get(tag, {})
        if not record:
            problems.append(f"no pinned revision for {tag}")
        elif record.get("status") != "PASS":
            problems.append(f"{tag} pre-flight status is {record.get('status')!r}")
        elif len(str(record.get("revision", ""))) != 40:
            problems.append(f"{tag} revision is not a full commit hash")
        if prereg and document.get("preregistration_sha256") != (
            sha_path.read_text(encoding="utf-8").split()[0]
        ):
            problems.append("revisions were pinned under a different preregistration")

    if (root / qc.results_path(tag)).exists():
        problems.append(
            f"{qc.results_path(tag)} already exists; a completed model is never rerun"
        )

    if problems:
        for problem in problems:
            print(f"GATE FAIL: {problem}", flush=True)
        raise SystemExit(2)
    return record


def _gpu_gib(peak: bool = False) -> float:
    try:
        import torch

        value = torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()
        return value / (1024**3)
    except Exception:  # pragma: no cover - operational reporting only
        return float("nan")


def append_receipt(root: Path, tag: str, receipt: dict) -> None:
    path = root / qc.receipts_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--tag", required=True, choices=list(qc.NEW_TAGS))
    parser.add_argument("--hf-token-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args()

    root, tag = Path(args.root), args.tag
    model_id = qc.MODEL_IDS[tag]
    record = gate(root, tag)
    revision = record["revision"]

    manifest = pd.read_parquet(root / d3c.PROMPT_MANIFEST_PATH).sort_values("order_index")
    if len(manifest) != N_ROWS:
        print(f"GATE FAIL: prompt manifest has {len(manifest)} rows", flush=True)
        return 2
    frozen_render = pd.read_parquet(root / qc.REFERENCE_RAW_SCORES_PATH).set_index(
        "prompt_id"
    )["rendered_prompt_sha256"]

    out_path = root / qc.results_path(tag)
    partial_path = root / qc.partial_results_path(tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if partial_path.exists():
        rows = pd.read_parquet(partial_path).to_dict("records")
    already = {row["prompt_id"] for row in rows}
    receipt = {
        "tag": tag,
        "model_id": model_id,
        "revision": revision,
        "invoked_at": provenance.utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "planned": int(len(manifest)),
        "already_scored_at_start": len(already),
    }

    token = None
    if args.hf_token_file and Path(args.hf_token_file).is_file():
        token = Path(args.hf_token_file).read_text(encoding="utf-8").strip() or None

    started = time.time()
    print(f"model={model_id}", flush=True)
    print(f"revision={revision}", flush=True)
    print(f"partition={qc.SOURCE_PARTITION}", flush=True)
    print(f"prompts={len(manifest)} already_scored={len(already)}", flush=True)
    print("loading model ...", flush=True)

    import torch
    from run_s3_inference import load_model

    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_model(model_id, revision, token, args.cache_dir)
    load_seconds = time.time() - started

    tokenization = scoring.resolve_label_tokenization(
        tokenizer, scoring.render_chat_prompt(tokenizer, prompts.label_reference_message())
    )
    chat_hash = provenance.sha256_text(tokenizer.chat_template or "")
    stop: list[str] = []
    if tokenization.to_dict() != record["label_tokenization"]:
        stop.append("label tokenisation differs from the pre-flight record")
    if tokenization.scoring_method != qc.REQUIRED_SCORING_METHOD:
        stop.append(f"scoring method is {tokenization.scoring_method}")
    if chat_hash != record["chat_template_sha256"]:
        stop.append("chat template hash differs from the pre-flight record")
    if str(next(model.parameters()).dtype) != f"torch.{MODEL_DTYPE}":
        stop.append("weights are not bfloat16")
    if not all(p.is_cuda for p in model.parameters()):
        stop.append("some parameters are not on the GPU")

    rendered_prompts: dict[str, str] = {}
    for entry in manifest.itertuples(index=False):
        rendered = scoring.render_chat_prompt(tokenizer, entry.user_message)
        if scoring.sha256_text(rendered) != frozen_render[entry.prompt_id]:
            stop.append(f"rendered prompt differs from the 14B record: {entry.prompt_id}")
            break
        rendered_prompts[entry.prompt_id] = rendered
    if stop:
        for problem in stop:
            print(f"GATE FAIL: {problem}", flush=True)
        append_receipt(root, tag, {**receipt, "outcome": "refused", "problems": stop})
        return 2

    failures: list[dict] = []
    newly_scored = 0
    print(f"loaded in {load_seconds:.1f}s; scoring ...", flush=True)

    scoring_started = time.time()
    for position, entry in enumerate(manifest.itertuples(index=False), start=1):
        if entry.prompt_id in already:
            continue
        rendered = rendered_prompts[entry.prompt_id]
        try:
            result = scoring.score_prompt(model, tokenizer, rendered, tokenization)
        except Exception as error:  # noqa: BLE001 - recorded, never hidden
            failures.append(
                {
                    "prompt_id": entry.prompt_id,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
            print(f"FAILURE at {entry.prompt_id}: {type(error).__name__}", flush=True)
            continue

        rows.append(
            {
                "prompt_id": entry.prompt_id,
                "artifact_id": entry.artifact_id,
                "family": entry.family,
                "latent_level": int(entry.latent_level),
                "strictness_index": int(entry.strictness_index),
                "order_index": int(entry.order_index),
                "partition": qc.SOURCE_PARTITION_TAG,
                "model_id": model_id,
                "revision": revision,
                "raw_score_met": result["raw_score_met"],
                "raw_score_not_met": result["raw_score_not_met"],
                "margin": result["raw_score_met"] - result["raw_score_not_met"],
                "p_met": result["p_met"],
                "scoring_method": result["scoring_method"],
                "n_prompt_tokens": result["n_prompt_tokens"],
                "rendered_prompt_sha256": result["rendered_prompt_sha256"],
            }
        )
        newly_scored += 1

        if position % args.checkpoint_every == 0:
            pd.DataFrame(rows, columns=RAW_SCORE_COLUMNS).to_parquet(
                partial_path, index=False
            )
            elapsed = time.time() - scoring_started
            rate = newly_scored / elapsed if elapsed else float("nan")
            print(
                f"progress {position}/{len(manifest)} elapsed={elapsed:.0f}s "
                f"rate={rate:.1f}/s gpu_alloc={_gpu_gib():.1f}GiB",
                flush=True,
            )

    frame = pd.DataFrame(rows, columns=RAW_SCORE_COLUMNS)
    complete = len(frame) == len(manifest) and frame["prompt_id"].is_unique
    if complete:
        frame = frame.sort_values("order_index").reset_index(drop=True)
        frame.to_parquet(out_path, index=False)
        if partial_path.exists():
            partial_path.unlink()
    else:
        frame.to_parquet(partial_path, index=False)

    total_seconds = time.time() - started
    scored_ids = set(frame["prompt_id"])
    failed_ids = {f["prompt_id"] for f in failures}
    completeness = {
        "planned": int(len(manifest)),
        "completed": int(len(frame)),
        "failed": int(len(failed_ids)),
        "missing": int(len(set(manifest["prompt_id"]) - scored_ids - failed_ids)),
        "scored_this_invocation": int(newly_scored),
        "carried_from_partial": int(len(already)),
    }
    append_receipt(
        root,
        tag,
        {
            **receipt,
            "outcome": "complete" if complete else "incomplete",
            "finished_at": provenance.utc_now(),
            **completeness,
            "failed_prompt_ids": sorted(failed_ids),
        },
    )
    receipts = [
        json.loads(line)
        for line in (root / qc.receipts_path(tag)).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    provenance.write_json(
        root / qc.run_record_path(tag),
        {
            "experiment": qc.STUDY_NAME,
            "partition": qc.SOURCE_PARTITION,
            "tag": tag,
            "model_id": model_id,
            "revision": revision,
            "tokenizer_revision": record["tokenizer_revision"],
            "dtype": MODEL_DTYPE,
            "quantization": "none",
            "batch_size": INFERENCE_BATCH_SIZE,
            "protocol": qc.PROTOCOL,
            "scoring_method": tokenization.scoring_method,
            "label_tokenization": tokenization.to_dict(),
            "chat_template_sha256": chat_hash,
            "prompt_template_sha256": prompts.prompt_template_sha256(),
            "dataset_file_sha256": provenance.sha256_file(root / d3c.DATASET_PATH),
            "prompt_manifest_file_sha256": provenance.sha256_file(
                root / d3c.PROMPT_MANIFEST_PATH
            ),
            "rendered_prompts_byte_identical_to_14b": True,
            "preregistration_sha256": (
                (root / qc.PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]
            ),
            "amendments_in_force": qs.effective_code_manifest(root)["n_amendments"],
            "model_revisions_sha256": provenance.sha256_file(
                root / qc.MODEL_REVISIONS_PATH
            ),
            "status": "COMPLETE" if complete else "INCOMPLETE",
            "completeness": completeness,
            "invocations": len(receipts),
            "retried": max(0, len(receipts) - 1),
            "failures": failures,
            "finished_at": provenance.utc_now(),
            "load_seconds": round(load_seconds, 2),
            "total_seconds": round(total_seconds, 2),
            "peak_gpu_memory_gib": round(_gpu_gib(peak=True), 3),
            "raw_scores_sha256": provenance.sha256_file(out_path) if complete else None,
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "environment": provenance.environment_snapshot(include_gpu=True),
            "source_manifest": provenance.source_manifest(PROTOCOL_SOURCE_PATHS, root=root),
        },
    )

    print(f"scored {len(frame)}/{len(manifest)} prompts", flush=True)
    print(f"failures: {len(failures)}", flush=True)
    print(f"total time: {total_seconds:.0f}s", flush=True)
    print(f"peak gpu: {_gpu_gib(peak=True):.1f}GiB", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
