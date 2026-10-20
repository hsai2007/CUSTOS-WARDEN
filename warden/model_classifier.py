"""M8: trained fast path (replaces the regex fact-gate).

`ModelFastPath` classifies every write with a fine-tuned DeBERTa-v3-small
cross-encoder (models/slow_stage_v3b, trained on Kaggle -- DECISIONS.md D25-D31).

Safety rule (D30): the model may only answer "update" when its update logit
beats its contradiction logit by at least `margin` (3.0, chosen on the dev
half before the test half was scored). A doubtful update is either
  * blocked as a contradiction (default, oracle=None), or
  * escalated to the LLM slow path (pass an oracle with .decide()).
Applying a false "update" writes a false fact into shared memory; blocking a
true one only costs a retry, so doubt resolves toward blocking.

torch / transformers are imported lazily, so the rest of WARDEN (and the
unit tests, which inject logits) runs without them installed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from warden.classifier import ClassificationResult
from warden.llm_client import PROJECT_ROOT
from warden.workload import CLASSES

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "slow_stage_v3b"
DEFAULT_MARGIN = 3.0


class ModelFastPath:
    def __init__(self, model_dir: Optional[Path] = None, margin: float = DEFAULT_MARGIN,
                 oracle=None, device: str = "cpu", max_length: int = 128):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.margin = margin
        self.oracle = oracle
        self.device = device
        self.max_length = max_length
        self._tok = None
        self._model = None
        self._labels: list[str] = []
        self._escalated = 0
        self._total = 0

    # -- model ---------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"no trained model at {self.model_dir}. Download slow_stage_v3b.zip from "
                f"Kaggle and extract it into models/ (see DECISIONS.md D31).")
        import torch  # noqa: F401  (lazy)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()
        id2label = self._model.config.id2label
        self._labels = [id2label[i] for i in range(len(id2label))]
        if sorted(self._labels) != sorted(CLASSES):
            raise ValueError(f"model labels {self._labels} do not match WARDEN classes {CLASSES}")

    def _logits(self, stored_text: str, incoming_text: str) -> dict[str, float]:
        """One forward pass -> {class: logit}. Tests override this."""
        self._load()                                   # fails loudly before touching torch
        import torch
        enc = self._tok(stored_text, incoming_text, truncation=True,
                        max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            row = self._model(**enc).logits[0].float().tolist()
        return dict(zip(self._labels, row))

    # -- classifier interface (same as FastPathEscalator) ----------------
    def classify(self, stored_text: str, incoming_text: str) -> ClassificationResult:
        start = time.perf_counter()
        self._total += 1
        logits = self._logits(stored_text, incoming_text)
        predicted = max(logits, key=logits.get)
        escalated = False

        if predicted == "update" and logits["update"] - logits["contradiction"] < self.margin:
            if self.oracle is None:
                predicted = "contradiction"          # doubtful update -> block
            else:
                escalated = True
                self._escalated += 1
                predicted = self.oracle.decide(stored_text, incoming_text)

        elapsed_ms = (time.perf_counter() - start) * 1000
        recorded = getattr(self.oracle, "last_latency_ms", None) if escalated else None
        if recorded is not None:                     # charge the real LLM latency, not cache replay (D23)
            elapsed_ms = max(elapsed_ms, recorded)
        return ClassificationResult(predicted, elapsed_ms, escalated)

    @property
    def escalation_rate(self) -> float:
        return self._escalated / self._total if self._total else 0.0
