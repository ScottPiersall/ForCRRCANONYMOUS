"""FormalCRRC: counterfactual rubric response curves under formal thresholds.

s 1 studies whether LLM judges reproduce the monotone counterfactual response
function implied by a rubric when the evaluated artifact is held byte-identical
and only a programmatically controlled decision threshold varies.

Ground truth in this package is computed by formal predicates, never by a model.
"""

__version__ = "0.1.0"

from formalcrrc import (  # noqa: F401
    bootstrap,
    config,
    dataset,
    s1_audit,
    s2,
    s2_bootstrap,
    s2_config,
    s3,
    s3_bootstrap,
    s3_config,
    metrics,
    predicates,
    prompts,
    provenance,
    scoring,
)

__all__ = [
    "__version__",
    "bootstrap",
    "config",
    "dataset",
    "s1_audit",
    "s2",
    "s2_bootstrap",
    "s2_config",
    "s3",
    "s3_bootstrap",
    "s3_config",
    "metrics",
    "predicates",
    "prompts",
    "provenance",
    "scoring",
]
