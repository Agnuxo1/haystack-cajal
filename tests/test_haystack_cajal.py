import unittest

from haystack_cajal import CAJALPaperGenerator, CAJALPaperReviewer


class HaystackCajalTests(unittest.TestCase):
    def test_generator_uses_injected_backend_and_counts_words(self) -> None:
        prompts: list[str] = []

        def backend(prompt: str) -> str:
            prompts.append(prompt)
            return "Evidence-backed draft text."

        generator = CAJALPaperGenerator(generator=backend)
        result = generator.run("photosynthesis", ["abstract", "methods"])
        self.assertEqual(list(result["paper"]), ["abstract", "methods"])
        self.assertEqual(result["word_count"], 6)
        self.assertEqual(len(prompts), 2)
        self.assertIn("Do not invent citations", prompts[0])

    def test_generator_rejects_invalid_sections(self) -> None:
        generator = CAJALPaperGenerator(generator=lambda _: "text")
        with self.assertRaises(ValueError):
            generator.run("topic", ["abstract", "abstract"])

    def test_reviewer_reports_transparent_heuristics(self) -> None:
        reviewer = CAJALPaperReviewer(experts=9)
        result = reviewer.run({
            "methodology": "We release code and data with a versioned protocol and DOI.",
            "limitations": "Limitations and ethics considerations are documented.",
        })
        self.assertEqual(result["report"]["rubric"], "deterministic-editorial-v1")
        self.assertTrue(result["report"]["human_review_required"])
        self.assertEqual(result["report"]["reviewer_count_metadata"], 9)
        self.assertIn(result["recommendation"], {"accept", "revise", "reject"})
        self.assertIn("reasons", result["paper"]["_review"])


if __name__ == "__main__":
    unittest.main()
