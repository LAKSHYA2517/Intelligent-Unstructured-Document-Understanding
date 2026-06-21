import json
from dataclasses import dataclass
from typing import List, Dict, Any
from retrieval import QueryAnalysis

@dataclass
class GeneratedAnswer:
    query: str
    answer_text: str
    citations: List[str]
    confidence: float
    grounding_status: dict

class MultiStageSynthesisEngine:
    SYNTHESIS_PROMPT_TEMPLATE = """You are an expert document analyst. 
Your task is to answer the question using ONLY the evidence provided.

EVIDENCE:
{evidence_context}

QUESTION: {question}

INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the evidence above.
2. For every factual claim you make, cite the evidence ID using the format [evidence_id].
3. If the evidence is INSUFFICIENT to fully answer the question, explicitly state which parts you cannot answer and why.
4. If evidence items CONTRADICT each other, acknowledge the contradiction and explain which evidence you are relying on and why.
5. Do NOT add information from your training data.
6. Format numbers consistently with how they appear in the evidence.

ANSWER FORMAT:
- Direct answer first (1-2 sentences)
- Supporting detail with citations [element_id]
- If partially unanswerable: "Note: I could not find evidence for [X]"

ANSWER:"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def synthesize(self, query: str, context: str, citation_ids: List[str], analysis: QueryAnalysis) -> GeneratedAnswer:
        if analysis.use_community and len(citation_ids) > 5:
            answer_text = self._map_reduce_synthesis(query, context, citation_ids)
        else:
            answer_text = self._single_pass_synthesis(query, context)

        used_citations = self._extract_used_citations(answer_text, citation_ids)
        grounding = self._check_grounding(answer_text, context)
        confidence = self._score_confidence(answer_text, grounding, analysis, len(used_citations))

        return GeneratedAnswer(
            query=query,
            answer_text=answer_text,
            citations=used_citations,
            confidence=confidence,
            grounding_status=grounding
        )

    def _single_pass_synthesis(self, query: str, context: str) -> str:
        prompt = self.SYNTHESIS_PROMPT_TEMPLATE.format(evidence_context=context, question=query)
        try:
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
        except Exception as e:
            content = f"Sorry, I encountered a network error while synthesizing the answer: {str(e)}"
        return content

    def _map_reduce_synthesis(self, query: str, context: str, citation_ids: List[str]) -> str:
        blocks = context.split('</evidence>')
        partials = []
        for i in range(0, len(blocks), 3):
            chunk = '</evidence>'.join(blocks[i:i+3])
            if not chunk.strip(): continue
            p = f"Based ONLY on this evidence, what can you say about: {query}\nEvidence:\n{chunk}\nProvide key facts only with citation IDs. If not addressed, say 'Not addressed here.'"
            res = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": p}],
                temperature=0.0
            )
            content = res.choices[0].message.content
            if "not addressed" not in content.lower():
                partials.append(content)
                
        if not partials:
            return "The available evidence does not contain sufficient information."
            
        combined = "\n\n---\n\n".join(f"[Partial {i+1}]:\n{p}" for i, p in enumerate(partials))
        reduce_prompt = f"Synthesize these partial analyses into ONE coherent answer for the question: {query}\nPartials:\n{combined}\nPreserve citation IDs."
        res = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": reduce_prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content

    def _extract_used_citations(self, answer_text: str, all_citations: List[str]) -> List[str]:
        import re
        cited = set(re.findall(r'\[([^\]]+)\]', answer_text))
        return [c for c in all_citations if c in cited]

    def _check_grounding(self, answer_text: str, context: str) -> dict:
        prompt = f"""Check if each factual claim in this answer is supported by the evidence.
ANSWER:
{answer_text}

EVIDENCE:
{context[:2000]}

Return JSON:
{{
  "overall_grounding_score": 0.9,
  "ungrounded_claims": []
}}"""
        try:
            res = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(res.choices[0].message.content)
        except:
            return {"overall_grounding_score": 0.5, "ungrounded_claims": []}

    def _score_confidence(self, answer_text: str, grounding: dict, analysis: QueryAnalysis, citation_count: int) -> float:
        score = grounding.get("overall_grounding_score", 0.5) * 0.4
        score += min(citation_count / 3.0, 1.0) * 0.2
        if "could not find" not in answer_text.lower():
            score += 0.2
        return round(max(0.0, min(1.0, score)), 3)
