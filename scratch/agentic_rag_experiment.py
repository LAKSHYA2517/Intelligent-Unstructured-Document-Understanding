import sys
import os
# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import json
from src.llm_router import agenerate
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph.falkordb_client import get_client

class QueryDecomposer:
    def __init__(self):
        self.retriever = HybridRetriever()

    async def decompose_query(self, query: str) -> dict:
        """Use the LLM to break down a complex multi-hop query into a DAG of sub-queries."""
        system_prompt = """You are a query decomposition module.
Your goal is to take a complex multi-hop question and break it into a logical Directed Acyclic Graph (DAG) of sub-queries.

Output valid JSON exactly in this format:
{
  "nodes": [
    {
      "id": "step_1",
      "type": "vector", // or "graph"
      "query": "Find the company that acquired Beta Labs.",
      "depends_on": []
    },
    {
      "id": "step_2",
      "type": "vector",
      "query": "What is the revenue of {step_1}?",
      "depends_on": ["step_1"]
    }
  ]
}
"""
        response = await agenerate(
            model="router-llm",
            system_prompt=system_prompt,
            prompt=query,
            json_mode=True
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON from LLM")
            return {"nodes": []}

    async def execute_plan(self, plan: dict):
        """Execute the DAG plan sequentially."""
        results = {}
        for node in plan.get("nodes", []):
            node_id = node["id"]
            node_type = node["type"]
            
            # Resolve dependencies in the query
            query = node["query"]
            for dep in node.get("depends_on", []):
                if dep in results:
                    # Inject the answer of the previous step into the prompt
                    query = query.replace(f"{{{dep}}}", str(results[dep]))
                    
            print(f"\\n--- Executing [{node_id}] ({node_type}) ---")
            print(f"Query: {query}")
            
            # Simple retrieval step
            if node_type == "vector":
                chunks = self.retriever.retrieve(query, top_k=3)
                # Synthesize a quick answer from the top chunks
                context = "\\n".join([c.text for c in chunks.chunks])
                
                synthesis_prompt = f"Context:\\n{context}\\n\\nAnswer the query concisely based ONLY on the context. If not found, say 'Unknown'.\\nQuery: {query}"
                ans = await agenerate(model="router-llm", prompt=synthesis_prompt)
                results[node_id] = ans.strip()
                print(f"Result: {ans.strip()}")
            elif node_type == "graph":
                # For graph, we might want the LLM to write a Cypher query, but for safety in this POC
                # we'll do a vector search but heavily boost entity exact matches.
                # (A full cypher generator could be added here!)
                chunks = self.retriever.retrieve(query, top_k=3)
                context = "\\n".join([c.text for c in chunks.chunks])
                synthesis_prompt = f"Context:\\n{context}\\n\\nAnswer the relational query concisely based ONLY on the context. If not found, say 'Unknown'.\\nQuery: {query}"
                ans = await agenerate(model="router-llm", prompt=synthesis_prompt)
                results[node_id] = ans.strip()
                print(f"Result: {ans.strip()}")
                
        return results

async def main():
    agent = QueryDecomposer()
    complex_query = "Who is the CEO of the company that acquired Acme Corp?"
    
    print(f"Original Query: {complex_query}")
    print("Decomposing...")
    plan = await agent.decompose_query(complex_query)
    
    print(json.dumps(plan, indent=2))
    
    print("\\nExecuting Plan...")
    final_results = await agent.execute_plan(plan)
    
    print("\\n=== FINAL STATE ===")
    for k, v in final_results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())
