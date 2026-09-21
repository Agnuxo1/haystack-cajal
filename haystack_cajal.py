"""Haystack 2.x components for local CAJAL-style paper workflows.

The generator uses an injected callable in tests and applications that already
own a model. The optional Transformers adapter is loaded lazily, so importing
the component never downloads weights or requires credentials. The reviewer is
an explicit, deterministic rubric; it does not impersonate peer review or
invent scores from a fixed table.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from haystack import component, default_from_dict, default_to_dict


DEFAULT_SECTIONS = ("abstract", "introduction", "methodology", "results", "conclusion")
REVIEW_DIMENSIONS = (
    "completeness",
    "clarity",
    "methodology",
    "reproducibility",
    "citation_presence",
    "limitations",
    "ethical_compliance",
    "coherence",
    "specificity",
)


def _normalise_sections(sections: Sequence[str] | None) -> list[str]:
    values = list(DEFAULT_SECTIONS if sections is None else sections)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("sections must contain at least one non-empty string")
    cleaned = [value.strip().lower().replace(" ", "_") for value in values]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("sections must not contain duplicates")
    return cleaned


def _word_count(text: str) -> int:
    return len(text.split())


@component
class CAJALPaperGenerator:
    """Generate section text through an injected backend or local Transformers.

    Parameters
    ----------
    model:
        Hugging Face model identifier or local model path used by the lazy
        Transformers adapter.
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``. The default never forces CUDA.
    generator:
        Optional callable accepting a prompt and returning text. Injecting one
        makes the component deterministic and keeps tests independent of model
        weights.
    """

    def __init__(
        self,
        model: str = "Agnuxo/CAJAL-4B-P2PCLAW",
        device: str = "auto",
        generator: Callable[[str], str] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        self.model = model
        self.device = device
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline: Any | None = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Local generation requires the 'local' extra: "
                "pip install haystack-cajal[local]"
            ) from exc

        pipeline_device = -1
        if self.device == "cuda":
            pipeline_device = 0
        elif self.device == "auto":
            try:
                import torch

                pipeline_device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                pipeline_device = -1
        self._pipeline = pipeline(
            "text-generation",
            model=self.model,
            device=pipeline_device,
        )
        return self._pipeline

    def _generate(self, prompt: str) -> str:
        if self.generator is not None:
            result = self.generator(prompt)
            if not isinstance(result, str) or not result.strip():
                raise ValueError("generator callable must return non-empty text")
            return result.strip()
        generator = self._load_pipeline()
        result = generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            return_full_text=False,
        )
        if not result or "generated_text" not in result[0]:
            raise RuntimeError("the local model returned no generated_text")
        text = result[0]["generated_text"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("the local model returned empty text")
        return text.strip()

    @component.output_types(paper=dict, word_count=int, generation_time=float)
    def run(self, topic: str, sections: Sequence[str] | None = None) -> dict[str, Any]:
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")
        selected_sections = _normalise_sections(sections)
        started = perf_counter()
        paper: dict[str, str] = {}
        for section in selected_sections:
            prompt = (
                "Write the {section} section of a scientific paper about {topic}. "
                "Return only the section text. Do not invent citations, results, "
                "measurements, participants, approvals or sources; mark missing "
                "evidence explicitly."
            ).format(section=section, topic=topic.strip())
            paper[section] = self._generate(prompt)
        elapsed = perf_counter() - started
        return {
            "paper": paper,
            "word_count": sum(_word_count(text) for text in paper.values()),
            "generation_time": elapsed,
        }

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model=self.model,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CAJALPaperGenerator":
        return default_from_dict(cls, data)


@dataclass(frozen=True)
class _DimensionResult:
    score: float
    reason: str


def _score_dimension(name: str, text: str, section_count: int) -> _DimensionResult:
    lowered = text.lower()
    words = text.split()
    if name == "completeness":
        score = min(10.0, 4.0 + section_count)
        reason = f"{section_count} paper section(s) supplied"
    elif name == "clarity":
        score = 8.0 if words and sum(len(word) for word in words) / len(words) < 8 else 6.0
        reason = "average word length is within the readability heuristic" if score == 8.0 else "review sentence and terminology complexity"
    elif name == "methodology":
        present = any(token in lowered for token in ("method", "protocol", "experiment", "dataset"))
        score = 8.0 if present else 4.0
        reason = "method-related language found" if present else "no method or protocol evidence found"
    elif name == "reproducibility":
        present = any(token in lowered for token in ("code", "data", "repository", "seed", "version"))
        score = 8.0 if present else 4.0
        reason = "reproducibility marker found" if present else "no reproducibility artifact marker found"
    elif name == "citation_presence":
        present = any(token in lowered for token in ("doi", "references", "citation", "[1]"))
        score = 8.0 if present else 3.0
        reason = "citation marker found" if present else "no citation marker found"
    elif name == "limitations":
        present = any(token in lowered for token in ("limitation", "uncertain", "unknown", "future work"))
        score = 8.0 if present else 4.0
        reason = "limitations or uncertainty are acknowledged" if present else "limitations are not explicitly acknowledged"
    elif name == "ethical_compliance":
        present = any(token in lowered for token in ("ethic", "consent", "privacy", "safety"))
        score = 8.0 if present else 5.0
        reason = "ethics or safety language found" if present else "no ethics or safety statement found"
    elif name == "coherence":
        score = 8.0 if len(words) >= 20 else 5.0
        reason = "section has enough text for a basic coherence check" if score == 8.0 else "section is too short for a strong coherence signal"
    elif name == "specificity":
        score = 8.0 if any(char.isdigit() for char in text) else 5.0
        reason = "quantitative or version detail found" if score == 8.0 else "no quantitative or version detail found"
    else:  # pragma: no cover - REVIEW_DIMENSIONS is closed above
        raise ValueError(f"unknown review dimension: {name}")
    return _DimensionResult(score, reason)


@component
class CAJALPaperReviewer:
    """Apply a transparent heuristic rubric to generated paper sections.

    The result is an editorial triage signal, not a substitute for qualified
    human peer review. ``experts`` is retained for API compatibility and is
    reported as metadata; no virtual reviewers or fabricated consensus are
    created.
    """

    def __init__(self, experts: int = 1) -> None:
        if experts < 1:
            raise ValueError("experts must be positive")
        self.experts = experts

    @component.output_types(scores=dict, paper=dict, recommendation=str, report=dict)
    def run(self, paper: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(paper, dict) or not paper:
            raise ValueError("paper must be a non-empty mapping of section names to text")
        if any(not isinstance(value, str) or not value.strip() for value in paper.values()):
            raise ValueError("paper section values must be non-empty strings")
        combined = "\n".join(paper.values())
        results = {
            name: _score_dimension(name, combined, len(paper)) for name in REVIEW_DIMENSIONS
        }
        scores = {name: result.score for name, result in results.items()}
        average = sum(scores.values()) / len(scores)
        recommendation = "accept" if average >= 7.5 and min(scores.values()) >= 6.0 else "revise" if average >= 5.0 else "reject"
        report = {
            "rubric": "deterministic-editorial-v1",
            "reviewer_count_metadata": self.experts,
            "average_score": round(average, 3),
            "reasons": {name: result.reason for name, result in results.items()},
            "human_review_required": True,
        }
        enriched = dict(paper)
        enriched["_review"] = report
        return {
            "scores": scores,
            "paper": enriched,
            "recommendation": recommendation,
            "report": report,
        }

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, experts=self.experts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CAJALPaperReviewer":
        return default_from_dict(cls, data)
