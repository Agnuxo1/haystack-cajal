"""Benchmark the deterministic component path without model weights."""

from __future__ import annotations

import json
from time import perf_counter

from haystack_cajal import CAJALPaperGenerator, CAJALPaperReviewer


def main() -> int:
    iterations = 100
    generator = CAJALPaperGenerator(generator=lambda _: "Evidence-backed draft text with protocol and DOI.")
    reviewer = CAJALPaperReviewer()
    started = perf_counter()
    successful = 0
    for _ in range(iterations):
        generated = generator.run("benchmark topic", ["abstract", "methodology"])
        reviewed = reviewer.run(generated["paper"])
        successful += reviewed["recommendation"] in {"accept", "revise", "reject"}
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    print(json.dumps({"iterations": iterations, "successful": successful, "elapsedMilliseconds": elapsed_ms}, sort_keys=True))
    return 0 if successful == iterations else 1


if __name__ == "__main__":
    raise SystemExit(main())
