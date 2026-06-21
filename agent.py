import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from memory import WorkingMemory, MemoryItem
from retrieval import ParallelHybridRetrievalEngine, QueryAnalysis, AdaptiveQueryPlanner

ACTIVE_RETRIEVAL_SYSTEM_PROMPT = """You are an expert document analyst with access 
to a typed knowledge graph. You reason carefully about what you know and don't know,
and retrieve specifically to fill gaps.

AVAILABLE TOOLS:
1. vector_search(query) -> semantic similarity search
2. graph_entity_search(entity) -> all elements connected to entity
3. table_search(query) -> specifically searches tables
4. community_search(topic) -> thematic summaries
5. compute(query, metric_ids) -> perform math on retrieved metrics
6. FINISH(answer) -> provide final answer

{memory_state}

Think step by step. What information are you missing? Which tool is best to find it?
Respond in JSON format:
{{
  "thought": "your reasoning",
  "action": "tool_name",
  "action_input": {{"key": "value"}}
}}
"""

@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: dict
    tool_result: Optional[dict] = None

@dataclass
class AgentResult:
    answer: str
    citations: List[str]
    confidence: float
    steps: List[AgentStep]
    total_iterations: int

class ComputationalReasoningTool:
    """Tool for auditing and computing on metrics."""
    def compute(self, operation: str, values: List[float]) -> Dict[str, Any]:
        try:
            if operation == "sum":
                return {"result": sum(values), "audit_trail": f"Summed {values}"}
            elif operation == "avg":
                return {"result": sum(values) / len(values) if values else 0, "audit_trail": f"Averaged {values}"}
            elif operation == "pct_change" and len(values) >= 2:
                v1, v2 = values[0], values[1]
                change = ((v2 - v1) / v1 * 100) if v1 != 0 else 0
                return {"result": change, "audit_trail": f"Pct change from {v1} to {v2}"}
            return {"error": "Unknown operation"}
        except Exception as e:
            return {"error": str(e)}

class ActiveRetrievalAgent:
    def __init__(self, retrieval_engine: ParallelHybridRetrievalEngine, query_planner: AdaptiveQueryPlanner, llm_client, max_iterations=6):
        self.retrieval = retrieval_engine
        self.planner = query_planner
        self.llm = llm_client
        self.memory = WorkingMemory()
        self.compute_tool = ComputationalReasoningTool()
        self.max_iterations = max_iterations

    def run(self, query: str, doc_id: Optional[str] = None) -> AgentResult:
        messages = []
        steps = []
        
        # Initial analysis to jump-start retrieval
        analysis = self.planner.analyze(query)
        initial_results = self.retrieval.search(query, analysis, doc_id)
        
        for res in initial_results[:3]:
            self.memory.add_fact(MemoryItem(
                element_id=res["element_id"],
                content=res["content"],
                source_type=res["retrieval_method"],
                confidence=res.get("rrf_score", 0.8),
                reason_for_inclusion="Initial hybrid retrieval"
            ))

        for iteration in range(self.max_iterations):
            system_prompt = ACTIVE_RETRIEVAL_SYSTEM_PROMPT.format(
                memory_state=self.memory.get_summary()
            )
            
            prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Question: {query}"}] + messages
            
            try:
                response = self.llm.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=prompt_messages,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            except Exception as e:
                logging.error(f"LLM call failed: {e}")
                return AgentResult(
                    answer=f"I encountered a network error while thinking: {str(e)}",
                    citations=[f.element_id for f in self.memory.short_term],
                    confidence=0.0,
                    steps=steps,
                    total_iterations=iteration + 1
                )
            
            try:
                agent_output = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                logging.error("Failed to parse LLM response as JSON")
                break
            except (AttributeError, IndexError):
                logging.error("Invalid response structure from LLM")
                break
                
            action = agent_output.get("action", "FINISH")
            action_input = agent_output.get("action_input", {})
            thought = agent_output.get("thought", "")
            
            if action == "FINISH" or action_input.get("answer"):
                return AgentResult(
                    answer=agent_output.get("answer", action_input.get("answer", "Insufficient evidence.")),
                    citations=[f.element_id for f in self.memory.short_term],
                    confidence=0.85,
                    steps=steps,
                    total_iterations=iteration + 1
                )
                
            tool_result = self._execute_tool(action, action_input, doc_id)
            
            steps.append(AgentStep(thought, action, action_input, tool_result))
            
            messages.append({"role": "assistant", "content": json.dumps(agent_output)})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_result)}"})

        return AgentResult(
            answer="Max iterations reached. Best effort answer based on memory.",
            citations=[f.element_id for f in self.memory.short_term],
            confidence=0.5,
            steps=steps,
            total_iterations=self.max_iterations
        )

    def _execute_tool(self, action: str, action_input: dict, doc_id: Optional[str]) -> dict:
        if action == "vector_search":
            q = action_input.get("query", "")
            results = self.retrieval._vector_search(self.retrieval._get_embedding(q), doc_id)
            return {"results": results[:3]}
        elif action == "graph_entity_search":
            ent = action_input.get("entity", "")
            results = self.retrieval._graph_entity_search(ent, depth=1, limit=5, doc_id=doc_id)
            return {"results": results}
        elif action == "compute":
            op = action_input.get("operation", "sum")
            vals = action_input.get("values", [])
            return self.compute_tool.compute(op, vals)
        return {"error": "Unknown tool"}
