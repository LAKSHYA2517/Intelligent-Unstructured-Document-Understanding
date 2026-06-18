from sentence_transformers import SentenceTransformer
import chromadb
import os

# Load the embedding model once
print("⏳ Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded!")

# Setup ChromaDB — runs locally, no server needed
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def store_chunks(chunks: list, filename: str) -> bool:
    try:
        # Create a collection named after the file
        collection_name = filename.replace(".pdf", "").replace(" ", "_").replace("-", "_")
        
        # Delete existing collection if file was uploaded before
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass
        
        collection = chroma_client.create_collection(name=collection_name)
        
        print(f"⏳ Embedding {len(chunks)} chunks...")
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            
            # Convert text to numbers
            embedding = model.encode(text).tolist()
            
            documents.append(text)
            embeddings.append(embedding)
            metadatas.append({
                "page": chunk["page"],
                "type": chunk["type"],
                "filename": filename
            })
            ids.append(f"chunk_{i}")
        
        # Store everything in ChromaDB
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Successfully stored {len(chunks)} chunks!")
        return True
        
    except Exception as e:
        print(f"❌ Error storing chunks: {e}")
        return False


def search_chunks(question: str, filename: str, top_k: int = 5) -> list:
    try:
        collection_name = filename.replace(".pdf", "").replace(" ", "_").replace("-", "_")
        collection = chroma_client.get_collection(name=collection_name)
        
        # Convert question to numbers
        question_embedding = model.encode(question).tolist()
        
        # Find most similar chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i]["page"],
                "type": results["metadatas"][0][i]["type"],
                "score": round(1 - results["distances"][0][i], 2)
            })
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return []


# ─── TEST ─────────────────────────────────────────
if __name__ == "__main__":
    from parser import parse_pdf
    
    test_file = "uploads/test.pdf"
    
    if os.path.exists(test_file):
        print("\n📄 Step 1: Parsing PDF...")
        chunks = parse_pdf(test_file)
        
        print("\n💾 Step 2: Storing in ChromaDB...")
        store_chunks(chunks, "test.pdf")
        
        print("\n🔍 Step 3: Testing search...")
        question = "what is this document about?"
        results = search_chunks(question, "test.pdf")
        
        print(f"\n📦 Top results for: '{question}'")
        for i, r in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Page: {r['page']}")
            print(f"  Type: {r['type']}")
            print(f"  Score: {r['score']}")
            print(f"  Text: {r['text'][:150]}...")
    else:
        print("❌ No test PDF found.")