import json
import logging
from typing import Optional

from knowledge import is_worth_extracting

logger = logging.getLogger(__name__)

BATCH_EXTRACTION_PROMPT = """Extract claims, metrics, and events from each 
document element below. Return a JSON object with a 'results' array containing one object per element,
in the SAME ORDER as the input.

Elements to analyze:
{elements_json}

Return ONLY a JSON object like this:
{{
  "results": [
    {{
    "element_id": "id_from_input",
    "claims": [
      {{
        "text": "declarative sentence stating the claim",
        "claim_type": "statistical|causal|comparative|definitional|predictive",
        "temporal_scope": "Q3 2024 or null",
        "entities_involved": ["entity names"],
        "numerical_values": [{{"value": 10.2, "unit": "billion USD"}}]
      }}
    ],
    "metrics": [
      {{
        "entity": "entity name",
        "metric_name": "revenue|gross_margin|etc",
        "value": 10.2,
        "unit": "USD_billion|percent|count",
        "period": "Q3_2024"
      }}
    ],
    "events": [
      {{
        "description": "what happened",
        "event_date": "2024-09-30 or null",
        "event_type": "earnings|acquisition|launch|regulatory",
        "entities_involved": ["names"]
      }}
    ]
  }}
  ]
}}

Rules:
- Return exactly one object per input element, in order
- Use empty arrays [] if nothing extractable
- element_id must match exactly what was given"""


class BatchKnowledgeExtractor:
    """
    Extracts knowledge from multiple document chunks in a single API call.
    Reduces API calls by 5-8x.
    """

    def __init__(
        self,
        managed_client,
        batch_size: int = 1,
        model: str = "llama-3.1-8b-instant"
    ):
        self.client     = managed_client
        self.batch_size = batch_size
        self.model      = model

    def extract_batch(self, elements: list) -> list[dict]:
        """
        Process a list of elements in batches.
        Returns a flat list of extraction results in the same order as input.
        """
        extractable = []
        non_extractable_ids = set()

        for element in elements:
            if is_worth_extracting(element):
                extractable.append(element)
            else:
                non_extractable_ids.add(element.element_id)

        logger.info(
            f"Batch extraction: {len(extractable)}/{len(elements)} elements "
            f"pass extraction filter "
            f"({len(non_extractable_ids)} skipped)"
        )

        results_by_id: dict[str, dict] = {}

        for batch_start in range(0, len(extractable), self.batch_size):
            batch = extractable[batch_start: batch_start + self.batch_size]

            batch_results = self._process_single_batch(batch)

            for result in batch_results:
                results_by_id[result["element_id"]] = result

            logger.debug(
                f"Processed batch {batch_start // self.batch_size + 1}/"
                f"{(len(extractable) + self.batch_size - 1) // self.batch_size}"
            )

        empty_result = {"claims": [], "metrics": [], "events": []}

        final_results = []
        for element in elements:
            if element.element_id in non_extractable_ids:
                final_results.append({
                    "element_id": element.element_id,
                    **empty_result
                })
            else:
                result = results_by_id.get(element.element_id, {
                    "element_id": element.element_id,
                    **empty_result
                })
                final_results.append(result)

        return final_results

    def _process_single_batch(self, batch: list) -> list[dict]:
        """Process one batch of elements with a single API call."""

        elements_for_prompt = [
            {
                "element_id":   el.element_id,
                "element_type": el.element_type,
                "content":      el.content[:600]
            }
            for el in batch
        ]

        prompt = BATCH_EXTRACTION_PROMPT.format(
            elements_json=json.dumps(elements_for_prompt, indent=2)
        )

        try:
            response = self.client.call(
                self.client.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1500
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Extract the array from the 'results' key
            results_list = parsed.get("results", [])
            
            # Validate format
            for res in results_list:
                if "element_id" not in res:
                    logger.warning(f"Extracted result missing element_id: {res}")

            if len(results_list) != len(batch):
                logger.warning(
                    f"Batch size mismatch: sent {len(batch)}, "
                    f"got {len(results_list)} results."
                )
                while len(results_list) < len(batch):
                    results_list.append({
                        "element_id": batch[len(results_list)].element_id,
                        "claims": [], "metrics": [], "events": []
                    })

            return results_list

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in batch: {e}")
            return [
                {
                    "element_id": el.element_id,
                    "claims": [], "metrics": [], "events": []
                }
                for el in batch
            ]

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            return [
                {
                    "element_id": el.element_id,
                    "claims": [], "metrics": [], "events": []
                }
                for el in batch
            ]
