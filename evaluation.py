from dataclasses import dataclass
from typing import Dict, Any

class SemanticEvaluationHarness:
    """
    Evaluation Harness to test the RAG Pipeline before the demo.
    Based on Layer 8: The Three Tests That Will Make or Break Your Demo
    """
    TEST_QUESTIONS = {
        "simple_factual": [
            "What is the total revenue for Q3 2024?",
            "Who is the CEO mentioned in this report?",
            "What page mentions the risk factors?"
        ],
        "cross_reference": [
            "The report mentions a footnote about revenue recognition. What does that footnote say?",
            "What does Section 3 say about the figures in Table 2?"
        ],
        "thematic": [
            "What are the main risks discussed in this report?",
            "Summarize the key financial highlights.",
            "What is the overall tone of management's commentary?"
        ],
        "multi_hop": [
            "How did the growth in revenue compare to the growth in expenses?",
            "What caused the change in operating margin between Q2 and Q3?"
        ],
        "negative_space": [
            "Does this report mention any supply chain disruptions?",
            "Are there any regulatory issues mentioned?"
        ],
        "conversational": [
            "What was Q3 revenue?",
            "How does that compare to Q2?",
            "What did management say about this trend?"
        ]
    }

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def run_all(self) -> Dict[str, Any]:
        results = {}

        for category, questions in self.TEST_QUESTIONS.items():
            category_results = []
            
            if category == "conversational":
                # Ensure sequential execution to test conversational memory
                for q in questions:
                    answer = self.pipeline.query(q)
                    category_results.append(self._format_result(q, answer))
            else:
                for q in questions:
                    answer = self.pipeline.query(q)
                    category_results.append(self._format_result(q, answer))

            results[category] = category_results

        return results

    def _format_result(self, question: str, answer) -> Dict[str, Any]:
        return {
            "question": question,
            "answer": answer.answer_text[:200],
            "confidence": answer.confidence,
            "citations": len(answer.citations),
            "grounded": answer.grounding_status.get("overall_grounding_score", 0.0) > 0.5
        }
