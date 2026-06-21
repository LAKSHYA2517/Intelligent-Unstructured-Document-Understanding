from typing import Optional, List, Dict
import os
import fitz
import base64
import ollama
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from knowledge import DocumentElement
from ingestion import IncrementalGraphBuilder
from retrieval import AdaptiveQueryPlanner, ParallelHybridRetrievalEngine
from agent import ActiveRetrievalAgent
from synthesis import MultiStageSynthesisEngine, GeneratedAnswer

class DocGraphRAGPipeline:
    """
    The complete DocGraph-RAG v3 Championship Architecture Pipeline.
    Combines: Knowledge Extraction, Incremental Ingestion, 
    Parallel Retrieval, Active Reasoning, and Synthesis.
    """
    def __init__(self, falkordb_host: str = "localhost", falkordb_port: int = 6379, qdrant_url: str = "http://localhost:6333"):
        # We assume local execution with Ollama.
        self.ollama_client = ollama.Client()
        
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("CRITICAL ERROR: GROQ_API_KEY environment variable is not set. Please set it in your .env file or terminal.")
            
        self.groq_client = Groq(
            api_key=api_key,
            timeout=120.0,
            max_retries=2
        )
        
        # We would initialize real db clients here
        self.falkordb_client = None 
        self.qdrant_client = None

        self.element_store: Dict[str, DocumentElement] = {}
        
        # 1. Ingestion Phase components
        self.graph_builder = IncrementalGraphBuilder(
            falkordb_client=self.falkordb_client,
            groq_client=self.groq_client,
            groq_vision_client=self.groq_client,
            cache_dir=".caption_cache"
        )
        
        # 2. Retrieval Phase components
        self.query_planner = AdaptiveQueryPlanner(self.groq_client)
        self.retrieval_engine = ParallelHybridRetrievalEngine(
            falkordb_client=self.falkordb_client,
            qdrant_client=self.qdrant_client,
            element_store=self.element_store,
            ollama_client=self.ollama_client  # Still used for nomic embeddings
        )
        
        # 3. Agent & Synthesis Phase components
        self.agent = ActiveRetrievalAgent(
            retrieval_engine=self.retrieval_engine,
            query_planner=self.query_planner,
            llm_client=self.groq_client
        )
        self.synthesizer = MultiStageSynthesisEngine(self.groq_client)
        
        self.conversation_history = []

    def ingest(self, file_path: str, doc_id: str) -> dict:
        """
        Parses document using PyMuPDF, structures it, and builds the knowledge graph.
        """
        elements = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text("text")
                if text and text.strip():
                    elements.append(DocumentElement(
                        element_id=f"{doc_id}_p{page_num}_text",
                        element_type="text",
                        content=text.strip()
                    ))
                    
                # Extract images
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        b64_img = base64.b64encode(image_bytes).decode('utf-8')
                        
                        elements.append(DocumentElement(
                            element_id=f"{doc_id}_p{page_num}_img{img_index}",
                            element_type="image",
                            content=b64_img
                        ))
                    except Exception as e:
                        print(f"Failed to extract image {img_index} on page {page_num}: {e}")
        except Exception as e:
            print(f"Failed to parse PDF {file_path}: {e}")
            raise e
        
        for element in elements:
            self.element_store[element.element_id] = element
            
        final_stats = {}
        for status in self.graph_builder.ingest_document(doc_id, elements):
            if status["phase"] == "complete":
                final_stats = status.get("stats", {})
                
        return {
            "doc_id": doc_id,
            "status": "ingested",
            "stats": final_stats
        }

    def query(self, question: str, doc_id: Optional[str] = None) -> GeneratedAnswer:
        """
        Orchestrates retrieval and generation using the Active Retrieval Agent.
        """
        analysis = self.query_planner.analyze(question)
        
        # The agent searches for evidence and reasons about it
        agent_result = self.agent.run(question, doc_id)
        
        # Assemble context from agent memory
        context_parts = []
        for elem_id in agent_result.citations:
            if elem_id in self.element_store:
                el = self.element_store[elem_id]
                context_parts.append(f"<evidence id='{elem_id}'>\n{el.content}\n</evidence>")
        
        final_context = "\n\n".join(context_parts)
        
        # Synthesize final answer using context and agent findings
        answer = self.synthesizer.synthesize(
            query=question,
            context=final_context,
            citation_ids=agent_result.citations,
            analysis=analysis
        )
        
        self.conversation_history.append({
            "question": question,
            "answer": answer.answer_text[:300]
        })
        
        return answer
