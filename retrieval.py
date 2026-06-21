import re
import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    original: str
    needs_number: bool
    needs_comparison: bool
    needs_list: bool
    needs_table: bool
    entity_mentions: List[str]
    time_references: List[str]
    hop_count: int
    use_vector: bool = True
    use_graph: bool = True
    use_community: bool = False
    use_table_filter: bool = False

class AdaptiveQueryPlanner:
    NUMBER_PATTERNS = [
        r'\bhow much\b', r'\bwhat (was|is|were) the\b', 
        r'\brevenue\b', r'\bprofit\b', r'\bgrowth\b',
        r'\bpercentage\b', r'\bnumber of\b', r'\bcount\b'
    ]
    COMPARISON_PATTERNS = [
        r'\bcompare\b', r'\bdifference\b', r'\bchange\b',
        r'\bincrease\b', r'\bdecrease\b', r'\bversus\b',
        r'\bvs\b', r'\byear.over.year\b', r'\bqoq\b', r'\byoy\b'
    ]
    THEMATIC_PATTERNS = [
        r'\bmain themes?\b', r'\bsummary\b', r'\boverall\b',
        r'\bwhat does this (report|document)\b', r'\bkey points?\b',
        r'\bmost important\b'
    ]
    TABLE_PATTERNS = [
        r'\btable\b', r'\bfigure\b', r'\bchart\b', r'\bshow me\b',
        r'\bdata\b', r'\bfinancials?\b', r'\bbalance sheet\b'
    ]

    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower()
        needs_number = any(re.search(p, query_lower) for p in self.NUMBER_PATTERNS)
        needs_comparison = any(re.search(p, query_lower) for p in self.COMPARISON_PATTERNS)
        use_community = any(re.search(p, query_lower) for p in self.THEMATIC_PATTERNS)
        use_table_filter = any(re.search(p, query_lower) for p in self.TABLE_PATTERNS)
        needs_list = bool(re.search(r'\blist\b|\bwhat are\b|\bwhat were\b', query_lower))
        
        # Simple entity extraction using LLM since spacy isn't guaranteed
        entity_mentions = self._extract_entities_llm(query)
        time_references = self._extract_times_llm(query)
        
        hop_count = 1
        multi_hop_signals = [
            "what does", "how does", "why did", "explain",
            "what caused", "what led to", "relationship between"
        ]
        if any(signal in query_lower for signal in multi_hop_signals):
            hop_count = 2

        return QueryAnalysis(
            original=query,
            needs_number=needs_number,
            needs_comparison=needs_comparison,
            needs_list=needs_list,
            needs_table=use_table_filter,
            entity_mentions=entity_mentions,
            time_references=time_references,
            hop_count=hop_count,
            use_vector=True,
            use_graph=len(entity_mentions) > 0,
            use_community=use_community,
            use_table_filter=use_table_filter
        )

    def _extract_entities_llm(self, query: str) -> List[str]:
        try:
            prompt = f"Extract named entities (organizations, people, products) from this query. Return a JSON object with an 'entities' array of strings. Query: {query}"
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content).get("entities", [])
        except:
            return []

    def _extract_times_llm(self, query: str) -> List[str]:
        try:
            prompt = f"Extract time references (dates, quarters, years) from this query. Return a JSON object with a 'times' array of strings. Query: {query}"
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content).get("times", [])
        except:
            return []

class ParallelHybridRetrievalEngine:
    def __init__(self, falkordb_client, qdrant_client, element_store, ollama_client):
        self.db = falkordb_client
        self.vectors = qdrant_client
        self.elements = element_store
        self.llm = ollama_client
        self.fusion = ReciprocalRankFusion()

    def search(self, query: str, analysis: QueryAnalysis, doc_id: Optional[str] = None) -> List[Dict]:
        path_results = {}
        
        # 1. Vector Search
        query_embedding = self._get_embedding(query)
        path_results["vector"] = self._vector_search(query_embedding, doc_id)
        
        # 2. Table Targeted Search
        if analysis.use_table_filter or analysis.needs_number:
            path_results["table_targeted"] = self._vector_search(query_embedding, doc_id, element_type="table")
            
        # 3. Graph Entity Search
        if analysis.use_graph and analysis.entity_mentions:
            graph_results = []
            for entity in analysis.entity_mentions[:3]:
                graph_results.extend(self._graph_entity_search(entity, depth=2, limit=10, doc_id=doc_id))
            path_results["graph_entity"] = graph_results
            
        # 4. Sequential Context
        if len(path_results.get("vector", [])) > 0:
            top_ids = [r["element_id"] for r in path_results["vector"][:5]]
            path_results["sequential_context"] = self._get_sequential_context(top_ids, window=1)
            
        # 5. Community Search
        if analysis.use_community:
            path_results["community"] = self._vector_search(query_embedding, doc_id, element_type="community")
            
        return self.fusion.fuse(path_results, analysis)

    def _get_embedding(self, text: str) -> List[float]:
        try:
            response = self.llm.embeddings(model="nomic-embed-text", prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.warning(f"Ollama embedding failed (is Ollama running?): {e}. Returning mock embedding.")
            return [0.0] * 768
        except:
            return [0.0] * 768

    def _vector_search(self, embedding: List[float], doc_id: Optional[str] = None) -> List[dict]:
        # Since Qdrant is mocked, we will return the first 5 elements for the doc_id from the element_store
        # so the system actually works for demo purposes.
        results = []
        for element in self.element_store.values():
            if doc_id and not element.element_id.startswith(doc_id):
                continue
            if element.element_type == "text":
                results.append({
                    "element_id": element.element_id,
                    "content": element.content,
                    "score": 0.85
                })
            if len(results) >= 5:
                break
        return results

    def _graph_entity_search(self, entity_text: str, depth: int, limit: int, doc_id: Optional[str]) -> List[Dict]:
        return []

    def _get_sequential_context(self, element_ids: List[str], window: int) -> List[Dict]:
        return []

class ReciprocalRankFusion:
    PATH_WEIGHTS = {
        "vector": 1.0,
        "table_targeted": 1.2,
        "graph_entity": 0.9,
        "sequential_context": 0.6,
        "community": 0.8,
    }

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, path_results: Dict[str, List[Dict]], analysis: QueryAnalysis) -> List[Dict]:
        adjusted_weights = self._adjust_weights_for_query(analysis)
        element_scores = {}
        element_data = {}
        element_paths = {}

        for path_name, results in path_results.items():
            weight = adjusted_weights.get(path_name, 0.8)
            for rank, result in enumerate(results):
                elem_id = result["element_id"]
                rrf_contribution = weight / (self.k + rank + 1)
                element_scores[elem_id] = element_scores.get(elem_id, 0.0) + rrf_contribution
                
                if elem_id not in element_data:
                    element_data[elem_id] = result.copy()
                element_paths.setdefault(elem_id, []).append(path_name)

        sorted_ids = sorted(element_scores.keys(), key=lambda eid: element_scores[eid], reverse=True)
        fused_results = []
        for elem_id in sorted_ids:
            result = element_data[elem_id].copy()
            result["rrf_score"] = element_scores[elem_id]
            result["paths_that_found_it"] = element_paths[elem_id]
            result["corroboration_count"] = len(set(element_paths[elem_id]))
            fused_results.append(result)

        return fused_results

    def _adjust_weights_for_query(self, analysis: QueryAnalysis) -> Dict[str, float]:
        weights = self.PATH_WEIGHTS.copy()
        if analysis.needs_number or analysis.needs_comparison:
            weights["table_targeted"] = 1.5
            weights["graph_entity"] = 1.1
        if len(analysis.entity_mentions) >= 2:
            weights["graph_entity"] = 1.3
        if analysis.use_community:
            weights["community"] = 1.4
            weights["vector"] = 0.7
        return weights
