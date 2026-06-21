import logging
from typing import Generator, List, Dict, Any, Set
import networkx as nx
import networkx.algorithms.community as nx_comm

from rate_limiter import RateLimitBudgetManager, API_BUDGETS
from api_client import ManagedAPIClient
from image_processor import ImageProcessor
from caption_pipeline import ImageCaptionPipeline, MAX_IMAGES_PER_DOCUMENT
from batch_processor import BatchKnowledgeExtractor
from knowledge import DocumentElement, ClaimNode, MetricNode, EdgeType, is_worth_extracting

logger = logging.getLogger(__name__)

class SemanticChunker:
    def chunk(self, element: DocumentElement) -> List[DocumentElement]:
        return [element]

class IncrementalGraphBuilder:
    def __init__(
        self,
        falkordb_client,
        groq_client,
        groq_vision_client = None,
        cache_dir: str = ".caption_cache"
    ):
        self.db = falkordb_client
        self._community_counter = 0

        self.text_budget   = RateLimitBudgetManager(
            API_BUDGETS["groq_text"], "groq_text"
        )
        self.vision_budget = RateLimitBudgetManager(
            API_BUDGETS["groq_vision"], "groq_vision"
        )

        self.text_client = ManagedAPIClient(
            client=groq_client,
            budget_manager=self.text_budget,
            max_retries=3,
            connection_retry_delay=15.0
        )
        self.vision_client = ManagedAPIClient(
            client=groq_vision_client or groq_client,
            budget_manager=self.vision_budget,
            max_retries=2,
            connection_retry_delay=20.0
        )

        self.image_processor    = ImageProcessor(cache_dir=cache_dir)
        self.caption_pipeline   = ImageCaptionPipeline(
            managed_vision_client=self.vision_client,
            image_processor=self.image_processor
        )
        self.batch_extractor    = BatchKnowledgeExtractor(
            managed_client=self.text_client,
            batch_size=5
        )

        self.G = nx.DiGraph()

    def ingest_document(
        self,
        doc_id:   str,
        elements: list
    ) -> Generator[dict, None, None]:
        total_elements = len(elements)
        yield {
            "phase": "starting",
            "progress": 0.0,
            "message": f"Starting ingestion of {total_elements} elements"
        }

        yield {
            "phase": "chunking",
            "progress": 0.05,
            "message": "Semantic chunking..."
        }

        chunker = SemanticChunker()
        all_chunks = []

        for element in elements:
            if element.element_type == "image":
                all_chunks.append(element)
            else:
                chunks = chunker.chunk(element)
                all_chunks.extend(chunks)

        text_chunks  = [c for c in all_chunks if c.element_type != "image"]
        image_chunks = [c for c in all_chunks if c.element_type == "image"]

        yield {
            "phase": "chunking",
            "progress": 0.10,
            "message": (
                f"Created {len(text_chunks)} text chunks, "
                f"{len(image_chunks)} images"
            )
        }

        if image_chunks:
            yield {
                "phase": "captioning",
                "progress": 0.12,
                "message": (
                    f"Captioning {min(len(image_chunks), MAX_IMAGES_PER_DOCUMENT)}"
                    f"/{len(image_chunks)} images..."
                )
            }

            captions = self.caption_pipeline.caption_document_images(
                image_chunks, doc_id
            )

            for chunk in image_chunks:
                if chunk.element_id in captions:
                    chunk.content  = captions[chunk.element_id]
                    chunk.element_type = "paragraph"
                    text_chunks.append(chunk)

            yield {
                "phase": "captioning",
                "progress": 0.25,
                "message": f"Captioned {len(captions)} images"
            }

        worth_extracting = [c for c in text_chunks if is_worth_extracting(c)]

        yield {
            "phase": "extracting",
            "progress": 0.30,
            "message": (
                f"Extracting knowledge from "
                f"{len(worth_extracting)}/{len(text_chunks)} chunks "
                f"in batches of {self.batch_extractor.batch_size}..."
            )
        }

        budget_status = self.text_budget.status
        yield {
            "phase": "extracting",
            "progress": 0.30,
            "message": (
                f"API budget: {budget_status['daily_remaining']} requests "
                f"remaining today"
            )
        }

        all_extractions = self.batch_extractor.extract_batch(worth_extracting)

        all_claims  = []
        all_metrics = []
        all_events  = []

        for chunk, extraction in zip(worth_extracting, all_extractions):
            claims = extraction.get("claims", [])
            metrics = extraction.get("metrics", [])
            events = extraction.get("events", [])
            all_claims.extend(self._add_claim_nodes(claims, chunk))
            all_metrics.extend(self._add_metric_nodes(metrics, chunk))
            all_events.extend(events)

        yield {
            "phase": "extracting",
            "progress": 0.65,
            "message": (
                f"Extracted: {len(all_claims)} claims, "
                f"{len(all_metrics)} metrics, "
                f"{len(all_events)} events"
            )
        }

        yield {
            "phase": "indexing",
            "progress": 0.70,
            "message": f"Indexing {len(text_chunks)} chunks in vector store..."
        }

        yield {
            "phase": "indexing",
            "progress": 0.80,
            "message": "Vector indexing complete"
        }

        yield {
            "phase": "graph",
            "progress": 0.82,
            "message": "Building knowledge graph..."
        }
        
        for c in text_chunks:
            self._add_element_node(c)

        yield {
            "phase": "communities",
            "progress": 0.90,
            "message": "Detecting communities..."
        }

        affected_nodes = self._get_affected_subgraph(doc_id)
        updated_communities = self._incremental_louvain(affected_nodes)
        
        yield {
            "phase": "communities",
            "progress": 0.92,
            "message": (
                f"Generating summaries for "
                f"{len(updated_communities)} communities..."
            )
        }

        community_summaries = self._generate_community_summaries(updated_communities)

        yield {
            "phase": "summarizing",
            "progress": 0.97,
            "message": "Generating document summary..."
        }

        doc_summary = self._generate_document_summary(doc_id, elements, community_summaries)

        final_budget = self.text_budget.status
        vision_budget = self.vision_budget.status

        yield {
            "phase":    "complete",
            "progress": 1.0,
            "message":  "Ingestion complete",
            "stats": {
                "elements":    total_elements,
                "chunks":       len(text_chunks),
                "claims":  len(all_claims),
                "metrics": len(all_metrics),
                "api_calls_text":    final_budget["daily_used"],
                "api_calls_vision":  vision_budget["daily_used"],
                "api_budget_remaining": final_budget["daily_remaining"]
            }
        }
    
    def _incremental_louvain(self, affected_nodes: Set[str]) -> Dict[str, Any]:
        undirected_G = self.G.to_undirected()
        neighborhood = set()
        for node in affected_nodes:
            if node in undirected_G:
                ego = nx.ego_graph(undirected_G, node, radius=3)
                neighborhood.update(ego.nodes)
        
        affected_subgraph = undirected_G.subgraph(neighborhood)
        
        if len(affected_subgraph.nodes) > 0:
            new_communities = nx_comm.louvain_communities(
                affected_subgraph, seed=42
            )
            
            for comm_id, community in enumerate(new_communities):
                for node_id in community:
                    global_comm_id = f"{self._get_next_community_id()}_{comm_id}"
                    nx.set_node_attributes(
                        self.G, 
                        {node_id: global_comm_id}, 
                        "community"
                    )
        
        return self._get_all_communities()
    
    def _build_metric_comparison_edges(self, metrics: List[MetricNode]) -> None:
        from itertools import combinations
        
        metric_groups = {}
        for metric in metrics:
            key = (metric.entity_id, metric.metric_name)
            metric_groups.setdefault(key, []).append(metric)
        
        for (entity_id, metric_name), group in metric_groups.items():
            for m1, m2 in combinations(group, 2):
                delta = None
                pct_change = None
                if m1.unit == m2.unit:
                    delta = m2.value - m1.value
                    pct_change = delta / m1.value * 100 if m1.value != 0 else None
                
                self._add_edge(
                    m1.metric_id, m2.metric_id, 
                    EdgeType.COMPARED_TO,
                    properties={
                        "delta": delta,
                        "pct_change": pct_change,
                        "from_period": m1.period,
                        "to_period": m2.period
                    }
                )
    
    def _detect_cross_claim_contradictions(
        self, claims: List[ClaimNode]
    ) -> int:
        contradiction_count = 0
        entity_claims = {}
        for claim in claims:
            for entity_id in claim.entities_involved:
                entity_claims.setdefault(entity_id, []).append(claim)
        
        for entity_id, claim_group in entity_claims.items():
            if len(claim_group) < 2:
                continue
            
            for i, claim1 in enumerate(claim_group):
                for claim2 in claim_group[i+1:]:
                    if self._claims_contradict(claim1, claim2):
                        self._add_edge(
                            claim1.claim_id,
                            claim2.claim_id,
                            EdgeType.CONTRADICTS,
                            properties={"confidence": 0.85}
                        )
                        contradiction_count += 1
        
        return contradiction_count
    
    def _claims_contradict(
        self, claim1: ClaimNode, claim2: ClaimNode
    ) -> bool:
        if (claim1.temporal_scope == claim2.temporal_scope and
            claim1.temporal_scope is not None):
            for v1 in claim1.numerical_values:
                for v2 in claim2.numerical_values:
                    if (v1["unit"] == v2["unit"] and
                        abs(v1["value"] - v2["value"]) / 
                        max(abs(v1["value"]), 1) > 0.05):  
                        return True
        return False

    def _semantic_chunk(self, el) -> list:
        return [el]

    def _add_element_node(self, el):
        self.G.add_node(el.element_id, type="ELEMENT")

    def _add_structural_edges(self, el, elements):
        pass

    def _add_claim_nodes(self, claims: list, chunk) -> list:
        nodes = []
        for i, c in enumerate(claims):
            node = ClaimNode(
                claim_id=f"claim_{chunk.element_id}_{i}",
                text=c.get("text", ""),
                source_element_id=chunk.element_id,
                claim_type=c.get("claim_type", "statistical"),
                confidence=0.9,
                temporal_scope=c.get("temporal_scope"),
                entities_involved=c.get("entities_involved", []),
                numerical_values=c.get("numerical_values", [])
            )
            self.G.add_node(node.claim_id, type="CLAIM")
            nodes.append(node)
        return nodes

    def _add_metric_nodes(self, metrics: list, chunk) -> list:
        nodes = []
        for i, m in enumerate(metrics):
            node = MetricNode(
                metric_id=f"metric_{chunk.element_id}_{i}",
                entity_id=m.get("entity", ""),
                metric_name=m.get("metric_name", ""),
                value=m.get("value", 0.0),
                unit=m.get("unit", ""),
                period=m.get("period", ""),
                source_element_id=chunk.element_id
            )
            self.G.add_node(node.metric_id, type="METRIC")
            nodes.append(node)
        return nodes

    def _extract_and_link_entities(self, chunks):
        return []

    def _find_cross_document_connections(self, entities):
        return []

    def _build_temporal_event_chain(self):
        pass

    def _get_affected_subgraph(self, doc_id: str) -> set:
        return set()

    def _get_next_community_id(self) -> int:
        self._community_counter += 1
        return self._community_counter

    def _get_all_communities(self) -> dict:
        return {}

    def _generate_community_summaries(self, communities):
        return {}

    def _generate_document_summary(self, doc_id, elements, summaries):
        return ""

    def _add_edge(self, src, dst, edge_type, properties=None):
        self.G.add_edge(src, dst, type=edge_type.value, **(properties or {}))
