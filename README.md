# haystack-cajal

Evidence-aware Haystack 2.x components for local CAJAL-style paper workflows.

This package provides two deliberately small components:

- `CAJALPaperGenerator` turns a topic and section list into prompts and delegates generation to an injected callable or a lazily loaded Hugging Face Transformers pipeline.
- `CAJALPaperReviewer` applies a deterministic editorial rubric to supplied text. It reports reasons and requires human review; it does not simulate reviewers or manufacture scientific consensus.

## Install

```bash
pip install haystack-cajal
```

For local Transformers inference, install the optional extra and provide model weights separately:

```bash
pip install haystack-cajal[local]
```

The package never downloads weights at import time and never requires an API key. Model licensing, hardware requirements and scientific validity remain the responsibility of the application owner.

## Haystack pipeline example

```python
from haystack import Pipeline
from haystack_cajal import CAJALPaperGenerator, CAJALPaperReviewer

pipeline = Pipeline()
pipeline.add_component("generator", CAJALPaperGenerator(model="Agnuxo/CAJAL-4B-P2PCLAW"))
pipeline.add_component("reviewer", CAJALPaperReviewer())
pipeline.connect("generator.paper", "reviewer.paper")

result = pipeline.run({
    "generator": {
        "topic": "Quantum effects in photosynthesis",
        "sections": ["abstract", "introduction", "methodology", "limitations"],
    }
})
print(result["reviewer"]["recommendation"])
```

For deterministic tests or an application-owned backend, inject a callable:

```python
generator = CAJALPaperGenerator(generator=lambda prompt: "Evidence-backed draft text.")
```

The generator explicitly instructs a model not to invent citations, measurements, participants, approvals or sources. The reviewer uses the `deterministic-editorial-v1` rubric and adds a report with reasons, average score and a `human_review_required` flag. Its scores are editorial triage signals, not peer review.

## Development

```bash
pip install -e ".[test]"
pytest
python -m compileall -q haystack_cajal.py tests
python scripts/benchmark.py
```

The benchmark exercises 100 injected-backend generations and reviewer runs. It measures component overhead only; it is not a model-quality or scientific-performance benchmark.

## Related projects and adapters

This integration targets [Haystack custom components](https://docs.haystack.deepset.ai/docs/custom-components), [Hugging Face Transformers](https://huggingface.co/docs/transformers/generation_strategies), the [CAJAL project](https://github.com/Agnuxo1/CAJAL), [P2PCLAW](https://github.com/Agnuxo1/P2PCLAW), and [LangGraph](https://github.com/langchain-ai/langgraph). These are documented connection points, not claims of upstream adoption or endorsement. The implementation is local-first and has no third-party network adapter.

## Limitations

- The local adapter requires compatible model weights and hardware; no weights are bundled.
- The deterministic reviewer cannot establish factual correctness, novelty, ethics approval or reproducibility.
- Human subject, clinical, safety and publication decisions require qualified human review.

## License

Apache License 2.0. See [LICENSE](LICENSE).
