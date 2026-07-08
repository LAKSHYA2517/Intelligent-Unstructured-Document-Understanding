**Day 1: Ingestion & Structuring (The Foundation)**

*   **Goal:** Turn a PDF into a clean, multi-modal dataset with UUIDs.
    
*   **Tasks:**
    
    *   Set up a Python script with Docling to parse the provided test PDFs.
        
    *   Write a function to intercept any extracted Image or Table, send it to the VLM API, and get a text description.
        
    *   Format the final output into a flat list of dictionaries, where every element has a chunk\_id, type (text/table/image), content, and parent\_header\_id.
        
*   **Checkpoint:** You can print a clean JSON payload representing the entire document layout.
    

**Day 2: Hybrid Memory & Graph Mapping (The Brain)**

*   **Goal:** Build the search and relationship engine.
    
*   **Tasks:**
    
    *   Initialize ChromaDB. Embed the content of every chunk and store it with its chunk\_id in the metadata.
        
    *   Initialize a NetworkX directed graph.
        
    *   Write the logic to iterate through your parsed JSON and add nodes and edges (e.g., Header\_1 $\\rightarrow$ Paragraph\_A, Paragraph\_A $\\rightarrow$ Table\_1).
        
    *   Wrap this ingestion process in a FastAPI /upload endpoint that stores the resulting NetworkX graph in app.state.
        
*   **Checkpoint:** You can query ChromaDB for a keyword, get a chunk\_id, and successfully print its neighboring nodes from NetworkX.
    

**Day 3: Agentic UI & Demo Polish (The Wow Factor)**

*   **Goal:** Connect the brain to the user and make it fast.
    
*   **Tasks:**
    
    *   Build the LangGraph agent: Query -> Vector DB (Find ID) -> NetworkX (Expand Context up to depth=1) -> LLM Prompt -> Answer.
        
    *   Implement StreamingResponse in FastAPI so the answer types out instantly.
        
    *   Build the React frontend. Implement the chat window on the left, and use React Flow on the right to visualize the nodes the AI traversed.
        
*   **Checkpoint:** End-to-end testing with sample Dell documents. Practice the pitch.
    

**Day 1: Ingestion & Structuring (The Foundation)**

*   **Goal:** Turn a PDF into a clean, multi-modal dataset with UUIDs.
    
*   **Tasks:**
    
    *   Set up a Python script with Docling to parse the provided test PDFs.
        
    *   Write a function to intercept any extracted Image or Table, send it to the VLM API, and get a text description.
        
    *   Format the final output into a flat list of dictionaries, where every element has a chunk\_id, type (text/table/image), content, and parent\_header\_id.
        
*   **Checkpoint:** You can print a clean JSON payload representing the entire document layout.
    

**Day 2: Hybrid Memory & Graph Mapping (The Brain)**

*   **Goal:** Build the search and relationship engine.
    
*   **Tasks:**
    
    *   Initialize ChromaDB. Embed the content of every chunk and store it with its chunk\_id in the metadata.
        
    *   Initialize a NetworkX directed graph.
        
    *   Write the logic to iterate through your parsed JSON and add nodes and edges (e.g., Header\_1 $\\rightarrow$ Paragraph\_A, Paragraph\_A $\\rightarrow$ Table\_1).
        
    *   Wrap this ingestion process in a FastAPI /upload endpoint that stores the resulting NetworkX graph in app.state.
        
*   **Checkpoint:** You can query ChromaDB for a keyword, get a chunk\_id, and successfully print its neighboring nodes from NetworkX.
    

**Day 3: Agentic UI & Demo Polish (The Wow Factor)**

*   **Goal:** Connect the brain to the user and make it fast.
    
*   **Tasks:**
    
    *   Build the LangGraph agent: Query -> Vector DB (Find ID) -> NetworkX (Expand Context up to depth=1) -> LLM Prompt -> Answer.
        
    *   Implement StreamingResponse in FastAPI so the answer types out instantly.
        
    *   Build the React frontend. Implement the chat window on the left, and use React Flow on the right to visualize the nodes the AI traversed.
        
*   **Checkpoint:** End-to-end testing with sample Dell documents. Practice the pitch.