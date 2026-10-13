#!/usr/bin/env python
"""Model availability gate and environment audit (Phase C).

Runs on a GPU compute node, before any FormalCRRC study prompt is scored. For
every preregistered judge model it:

1. verifies repository and tokenizer access,
2. resolves the exact repository revision,
3. inspects how the tokenizer encodes the answer labels ``A`` and ``B`` and
   records which preregistered scoring rule that implies,
4. loads the model in bf16 and runs a **non-study** smoke prompt,
5. records dtype, library versions, GPU and chat-template hashes.

The smoke prompt deliberately has nothing to do with FormalCRRC: no rubric, no
threshold, no candidate response. A model that cannot be reached is recorded as
``BLOCKED_BY_MODEL_ACCESS``; nothing is silently substituted.

Writes ``artifacts/s1/model_provenance.json``.

Secrets: a Hugging Face token is read from a file path given by
``--hf-token-file`` and passed straight to the library. It is never printed,
logged, or written to any artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import prompts, provenance, scoring  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    MODEL_DTYPE,
    MODEL_IDS,
    MODEL_PROVENANCE_PATH,
    MODEL_SLUGS,
)

#: A neutral prompt with no connection to the study design.
SMOKE_PROMPT = "Name one primary colour."

STATUS_AVAILABLE = "AVAILABLE"
STATUS_BLOCKED = "BLOCKED_BY_MODEL_ACCESS"


def read_token(path: str | None) -> str | None:
    """Read a Hugging Face token from disk without ever echoing it."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        print(f"note: no token file at {path}", flush=True)
        return None
    return candidate.read_text(encoding="utf-8").strip() or None


def resolve_revision(model_id: str, token: str | None) -> str:
    """Resolve the model repository's exact commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, token=token)
    return info.sha


def audit_model(model_id: str, token: str | None, cache_dir: str | None) -> dict:
    """Full availability check and provenance record for one judge model."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    record: dict = {
        "model_id": model_id,
        "slug": MODEL_SLUGS[model_id],
        "status": STATUS_BLOCKED,
        "checked_at": provenance.utc_now(),
    }

    revision = resolve_revision(model_id, token)
    record["revision"] = revision
    record["tokenizer_revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token, cache_dir=cache_dir
    )
    # Labels are resolved against a structurally identical prompt that carries no
    # study content, and by tokenizer inspection alone -- no model call.
    label_reference = scoring.render_chat_prompt(
        tokenizer, prompts.label_reference_message()
    )
    tokenization = scoring.resolve_label_tokenization(tokenizer, label_reference)
    record["label_tokenization"] = tokenization.to_dict()
    record["label_reference_prompt_sha256"] = provenance.sha256_text(label_reference)
    record["chat_template_sha256"] = provenance.sha256_text(
        tokenizer.chat_template or ""
    )
    record["tokenizer"] = {
        "class": type(tokenizer).__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
    }
    rendered_probe = scoring.render_chat_prompt(tokenizer, SMOKE_PROMPT)
    record["smoke_rendered_prompt_sha256"] = provenance.sha256_text(rendered_probe)

    # Chat templates usually emit their own BOS. Encoding with
    # add_special_tokens=False must therefore not add a second one; record the
    # leading tokens so a duplicate would be visible in provenance.
    probe_ids = tokenizer.encode(rendered_probe, add_special_tokens=False)
    leading = [tokenizer.decode([i]) for i in probe_ids[:4]]
    bos = tokenizer.bos_token
    record["prompt_encoding"] = {
        "add_special_tokens": False,
        "n_tokens": len(probe_ids),
        "leading_token_ids": [int(i) for i in probe_ids[:4]],
        "leading_tokens_decoded": leading,
        "bos_token": bos,
        "bos_duplicated": bool(
            bos is not None and len(leading) >= 2 and leading[0] == bos == leading[1]
        ),
    }

    dtype = getattr(torch, MODEL_DTYPE)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, token=token, cache_dir=cache_dir
        )
    except TypeError:
        # transformers < 5 spells the argument torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            token=token,
            cache_dir=cache_dir,
        )
    model = model.to("cuda").eval()

    rendered = scoring.render_chat_prompt(tokenizer, SMOKE_PROMPT)
    smoke = scoring.score_prompt(model, tokenizer, rendered, tokenization)

    record["dtype"] = str(next(model.parameters()).dtype)
    record["model_config"] = {
        "architectures": getattr(model.config, "architectures", None),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "max_position_embeddings": getattr(
            model.config, "max_position_embeddings", None
        ),
    }
    record["smoke_test"] = {
        "prompt": SMOKE_PROMPT,
        "is_study_prompt": False,
        "raw_score_met": smoke["raw_score_met"],
        "raw_score_not_met": smoke["raw_score_not_met"],
        "scoring_method": smoke["scoring_method"],
        "finite": True,
    }
    record["transformers_version"] = transformers.__version__
    record["torch_version"] = torch.__version__
    record["status"] = STATUS_AVAILABLE

    del model
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--hf-token-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--models", nargs="*", default=list(MODEL_IDS), help="subset to audit"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge into an existing model_provenance.json instead of replacing it",
    )
    args = parser.parse_args()

    root = Path(args.root)
    token = read_token(args.hf_token_file)
    out_path = root / MODEL_PROVENANCE_PATH

    models: dict = {}
    if args.merge and out_path.is_file():
        models = json.loads(out_path.read_text(encoding="utf-8")).get("models", {})

    for model_id in args.models:
        print(f"--- auditing {model_id}", flush=True)
        try:
            record = audit_model(model_id, token, args.cache_dir)
        except Exception as error:  # noqa: BLE001 - recorded, never silently swallowed
            record = {
                "model_id": model_id,
                "slug": MODEL_SLUGS.get(model_id, model_id.replace("/", "_")),
                "status": STATUS_BLOCKED,
                "checked_at": provenance.utc_now(),
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
                "traceback_tail": traceback.format_exc()[-2000:],
            }
            print(f"    {STATUS_BLOCKED}: {type(error).__name__}", flush=True)
        else:
            method = record["label_tokenization"]["scoring_method"]
            print(
                f"    {STATUS_AVAILABLE} revision={record['revision'][:12]} "
                f"scoring={method}",
                flush=True,
            )
        models[model_id] = record

    document = {
        "created_at": provenance.utc_now(),
        "preregistered_panel": list(MODEL_IDS),
        "dtype_requested": MODEL_DTYPE,
        "smoke_prompt": SMOKE_PROMPT,
        "smoke_prompt_is_study_prompt": False,
        "environment": provenance.environment_snapshot(include_gpu=True),
        "models": models,
        "panel_status": {
            model_id: models.get(model_id, {}).get("status", "NOT_CHECKED")
            for model_id in MODEL_IDS
        },
    }
    provenance.write_json(out_path, document)

    available = sum(
        1 for m in MODEL_IDS if models.get(m, {}).get("status") == STATUS_AVAILABLE
    )
    print(f"available: {available}/{len(MODEL_IDS)}", flush=True)
    print(f"wrote {MODEL_PROVENANCE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
