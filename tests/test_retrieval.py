import unittest

from resume_rag.generation import build_prompt
from resume_rag.retrieval import RetrievedChunk, retrieve_context


class RetrieveContextTests(unittest.TestCase):
    def test_retrieves_relevant_resume_evidence(self):
        resume = (
            "Built Python machine learning pipelines and deployed NLP models. "
            "Managed office schedules and organized meetings."
        )
        job = "Seeking a machine learning engineer with Python and NLP experience."

        results = retrieve_context(resume, job, top_k=4)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].source, "Resume evidence")
        self.assertIn("Python", results[0].text)
        self.assertGreater(results[0].score, 0)

    def test_rejects_empty_inputs_and_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "Resume text"):
            retrieve_context(" ", "Data scientist")
        with self.assertRaisesRegex(ValueError, "Job description"):
            retrieve_context("Python engineer", "")
        with self.assertRaisesRegex(ValueError, "top_k"):
            retrieve_context("Python engineer", "Python role", top_k=0)


class BuildPromptTests(unittest.TestCase):
    def test_prompt_includes_context_and_factuality_constraints(self):
        chunk = RetrievedChunk(
            source="Resume evidence",
            text="Built Python data pipelines.",
            score=0.8,
        )

        prompt = build_prompt(
            "Data engineer with Python experience.",
            "Seeking Python and SQL.",
            [chunk],
        )

        self.assertIn("Do not invent", prompt)
        self.assertIn("Built Python data pipelines.", prompt)
        self.assertIn("Seeking Python and SQL.", prompt)
        self.assertIn("Gaps to Address", prompt)


if __name__ == "__main__":
    unittest.main()
