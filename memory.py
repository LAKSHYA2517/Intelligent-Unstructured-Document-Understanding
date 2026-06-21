from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class MemoryItem:
    element_id: str
    content: str
    source_type: str
    confidence: float
    reason_for_inclusion: str

class WorkingMemory:
    """
    3-Layer Working Memory for the ActiveRetrievalAgent
    """
    def __init__(self):
        self.short_term: List[MemoryItem] = []          # Recent retrieved facts
        self.graph_path: List[str] = []                 # Nodes traversed so far
        self.contradictions: List[Dict[str, Any]] = []  # Detected conflicts
        self.max_capacity = 10

    def add_fact(self, item: MemoryItem):
        self.short_term.append(item)
        if len(self.short_term) > self.max_capacity:
            # Simple FIFO eviction, could be improved with semantic eviction
            self.short_term.pop(0)

    def record_traversal(self, element_id: str):
        if element_id not in self.graph_path:
            self.graph_path.append(element_id)

    def register_contradiction(self, claim1_id: str, claim2_id: str, desc: str):
        self.contradictions.append({
            "claim1": claim1_id,
            "claim2": claim2_id,
            "description": desc
        })

    def get_summary(self) -> str:
        summary = "CURRENT KNOWLEDGE STATE:\n"
        summary += "- FACTS GATHERED:\n"
        for i, item in enumerate(self.short_term):
            summary += f"  {i+1}. [{item.element_id}] {item.content} (Source: {item.source_type})\n"
        
        if self.contradictions:
            summary += "\n- CONTRADICTIONS DETECTED:\n"
            for c in self.contradictions:
                summary += f"  * Conflict between {c['claim1']} and {c['claim2']}: {c['description']}\n"
                
        return summary
