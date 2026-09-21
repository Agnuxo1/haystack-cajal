# Haystack Custom Component for CAJAL

This is a Haystack 2.x custom component that integrates CAJAL (local scientific paper generator) into Haystack pipelines.

## Installation

```bash
pip install haystack-cajal
```

## Usage

```python
from haystack import Pipeline
from haystack_cajal import CAJALPaperGenerator, CAJALPaperReviewer

# Create pipeline
pipeline = Pipeline()
pipeline.add_component("generator", CAJALPaperGenerator(model="Agnuxo/CAJAL-4B-P2PCLAW"))
pipeline.add_component("reviewer", CAJALPaperReviewer(experts=9))

# Run
result = pipeline.run({
    "generator": {"topic": "Quantum effects in photosynthesis", "sections": ["abstract", "intro", "methods", "results", "conclusion"]}
})

# Output: paper with AI Tribunal scores
paper = result["reviewer"]["paper"]
scores = result["reviewer"]["scores"]
```

## Features

- **Local execution** — No API keys, runs on your hardware
- **Structured output** — Guaranteed paper sections via Haystack output validation
- **AI Tribunal** — 9 virtual reviewers score each section
- **PDF export** — Direct integration with Haystack document stores

## Components

### CAJALPaperGenerator
Generates scientific papers from a topic description.

**Input:**
- `topic` (str): Research topic
- `sections` (List[str]): Paper sections to generate

**Output:**
- `paper` (dict): Structured paper with all sections
- `word_count` (int): Total word count
- `generation_time` (float): Time to generate

### CAJALPaperReviewer
AI Tribunal peer review for generated papers.

**Input:**
- `paper` (dict): Paper from generator

**Output:**
- `scores` (dict): 9-dimension scoring
- `paper` (dict): Paper with scores embedded
- `recommendation` (str): "accept", "revise", or "reject"

## License
Apache 2.0

## Links
- https://github.com/Agnuxo1/CAJAL
- https://huggingface.co/Agnuxo/CAJAL-4B-P2PCLAW
- https://www.p2pclaw.com/silicon
